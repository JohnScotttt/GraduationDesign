from dataclasses import dataclass


@dataclass
class settings_params:
    epochs: int
    batch_size: int
    device: str
    num_workers: int
    train_tsv_file: str
    eval_tsv_file: str
    output_dir: str
    name: str
    ckpt: str
    early_stop: int
    conditional: bool
    sampling_timesteps: int
    weight: list[float]

@dataclass
class detail_params:
    optimizer: str
    scheduler: str
    lr: float
    weight_decay: float
    momentum: float
    factor: float
    patience: int
    min_lr: float
    step_size: int
    in_channels: int
    base_channels: int
    transformer_dim: int
    patch_size: int
    num_transformer_layers: int
    num_heads: int


@dataclass
class diffusion_params:
    optimizer: str
    scheduler: str
    weight_decay: float
    lr: float
    momentum: float
    factor: float
    patience: int
    min_lr: float
    amsgrad: bool
    eps: float
    step_size: int
    gamma: float
    beta_schedule: str
    beta_start: float
    beta_end: float
    num_diffusion_timesteps: int
    in_channels: int
    out_ch: int
    ch: int
    ch_mult: list[int]
    num_res_blocks: int
    dropout: float
    ema_rate: float
    ema: bool
    resamp_with_conv: bool


@dataclass
class loss_params:
    epsilon: float
    lambda_vgg: float


@dataclass
class params:
    settings: settings_params
    detail: detail_params
    diffusion: diffusion_params
    loss: loss_params

    def __post_init__(self):
        self.settings = settings_params(**self.settings)
        self.detail = detail_params(**self.detail)
        self.diffusion = diffusion_params(**self.diffusion)
        self.loss = loss_params(**self.loss)
