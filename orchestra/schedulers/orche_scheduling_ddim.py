from diffusers.schedulers.scheduling_ddim import (
    DDIMScheduler, 
    DDIMSchedulerOutput
)
from diffusers.configuration_utils import register_to_config
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ..config import OrcheConfig
from ..coordinator import OrcheCoordinator

class OrcheDDIMScheduler(DDIMScheduler):

    
    @register_to_config
    def __init__(self, 
                 orche_config: OrcheConfig=None, 
                 num_train_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 beta_schedule: str = "linear",
                 trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
                 clip_sample: bool = True,
                 set_alpha_to_one: bool = True,
                 steps_offset: int = 0,
                 prediction_type: str = "epsilon",
                 thresholding: bool = False,
                 dynamic_thresholding_ratio: float = 0.995,
                 clip_sample_range: float = 1.0,
                 sample_max_value: float = 1.0,
                 timestep_spacing: str = "leading",
                 rescale_betas_zero_snr: bool = False,
                 ):
        super().__init__(
                 num_train_timesteps,
                 beta_start,
                 beta_end,
                 beta_schedule,
                 trained_betas,
                 clip_sample,
                 set_alpha_to_one,
                 steps_offset,
                 prediction_type,
                 thresholding,
                 dynamic_thresholding_ratio,
                 clip_sample_range,
                 sample_max_value,
                 timestep_spacing,
                 rescale_betas_zero_snr)

        self.orche_config = orche_config
        self.rank = orche_config.split_idx()

        self.warmup_steps = orche_config.warmup_steps 
        self.base_num_inference_steps = orche_config.num_inference_steps 
        self.warmup_equivalent_steps = orche_config.num_inference_steps 
        self.regular_equivalent_steps = None
        
        self.coordinator = None

        self.timesteps_cache = None
        self.adjusted_alphas_cumprod = None 

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        subfolder: Optional[str] = None,
        return_unused_kwargs=False,
        **kwargs,
    ):
        config, kwargs, commit_hash = cls.load_config(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            subfolder=subfolder,
            return_unused_kwargs=True,
            return_commit_hash=True,
            **kwargs,
        )
        scheduler = cls.from_config(config, return_unused_kwargs=return_unused_kwargs, **kwargs)
        return scheduler
    
    def set_coordinator(self, orche_coordinator: OrcheCoordinator):
        self.coordinator = orche_coordinator
        self.comm_interval = orche_coordinator.comm_interval
        
        if (self.orche_config.split_batch and 
            self.orche_config.do_classifier_free_guidance and 
            self.orche_config.world_size >= 2):
            raise NotImplementedError("We have not yet adapted it for batch splitting technology.")
        else:
            self.regular_equivalent_steps = orche_coordinator.all_time_steps[self.rank]

    def _generate_timesteps_by_period(
        self, 
        equivalent_steps,
        num_train_timesteps,
        start_step,
        end_step,
        timestep_spacing: str,
    ):
        if timestep_spacing == "leading":
            step_ratio = num_train_timesteps // equivalent_steps
            assert (start_step-end_step)%step_ratio == 0, "Illegal period"
            
            actual_steps = (start_step - end_step) // step_ratio
            timesteps = ((np.arange(0, actual_steps) * step_ratio).round()[::-1]+end_step).copy().astype(np.int64)
            assert timesteps[-1] == end_step
        else:
            raise ValueError(f"Unsupported timestep_spacing: {timestep_spacing}")
        return timesteps

    def _generate_timesteps(
        self,
        num_inference_steps: int,
        num_train_timesteps: int,
        timestep_spacing: str,
        steps_offset: int = 0,
    ) -> np.ndarray:
        if num_inference_steps > num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `num_train_timesteps`:"
                f" {num_train_timesteps}."
            )
        
        if timestep_spacing == "leading":
            step_ratio = num_train_timesteps // num_inference_steps
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            timesteps += steps_offset
        else:
            raise ValueError(f"Unsupported timestep_spacing: {timestep_spacing}")

        return timesteps
    
    def set_timesteps(self, num_inference_steps: int = None, device: Union[str, torch.device]  = None):
        assert self.coordinator is not None, "'set_coordinator()' must be called before 'set_timesteps()'"
        assert self.coordinator.all_time_steps is not None, "all_time_steps is None"
        num_inference_steps = num_inference_steps if num_inference_steps is not None else self.regular_equivalent_steps
        
        if self.warmup_steps > self.base_num_inference_steps:
            raise ValueError(
                f"warmup_steps {self.warmup_steps} > {self.base_num_inference_steps}"
            )
        
        if num_inference_steps == self.base_num_inference_steps:
            self.timesteps = self._generate_timesteps(
                num_inference_steps=num_inference_steps,
                num_train_timesteps=self.config.num_train_timesteps,
                timestep_spacing=self.config.timestep_spacing,
                steps_offset=self.config.steps_offset
            )
        else:
            warmup_offset = self.warmup_steps % 2 
            
            warmup_equivalent_timesteps = self._generate_timesteps(
                num_inference_steps=self.warmup_equivalent_steps,
                num_train_timesteps=self.config.num_train_timesteps,
                timestep_spacing=self.config.timestep_spacing,
                steps_offset=self.config.steps_offset
            )
            warmup_timesteps = warmup_equivalent_timesteps[:self.warmup_steps]
            offset_timesteps = warmup_equivalent_timesteps[-(warmup_offset+1):]

            regular_timesteps = self._generate_timesteps_by_period(
                equivalent_steps=num_inference_steps,
                start_step=warmup_timesteps[-1],
                end_step=offset_timesteps[0],
                num_train_timesteps=self.config.num_train_timesteps,
                timestep_spacing=self.config.timestep_spacing,
            )
            self.timesteps = np.unique(np.concatenate([warmup_timesteps, regular_timesteps, offset_timesteps]))[::-1]

        self.timesteps_cache = self.timesteps.tolist()
        self.adjusted_alphas_cumprod = torch.tensor(
            [self.alphas_cumprod[int(t)] for t in self.timesteps], 
            device=device
        )

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        step_index = self.timesteps_cache.index(timestep)
        prev_timestep = self.timesteps_cache[step_index + 1] if step_index + 1 < len(self.timesteps_cache) else -1

        alpha_prod_t = self.adjusted_alphas_cumprod[step_index]
        alpha_prod_t_prev = (
            self.adjusted_alphas_cumprod[step_index + 1] 
            if step_index + 1 < len(self.adjusted_alphas_cumprod) 
            else self.final_alpha_cumprod
        )

        beta_prod_t = 1 - alpha_prod_t

        if self.config.prediction_type == "epsilon": # SDXL
            pred_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t **0.5
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t**0.5 * pred_original_sample) / beta_prod_t** 0.5
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample

        if self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-self.config.clip_sample_range, self.config.clip_sample_range)

        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t = eta * variance ** 0.5

        if use_clipped_model_output: 
            pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t**0.5

        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** 0.5 * pred_epsilon

        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

        if eta > 0:
            noise = variance_noise if variance_noise is not None else torch.randn_like(model_output)
            prev_sample = prev_sample + std_dev_t * noise

        if not return_dict:
            return (prev_sample,)

        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample) if return_dict else (prev_sample,)

