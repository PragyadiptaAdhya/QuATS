from pathlib import Path
from torch import nn
from peft import LoraConfig, get_peft_model
from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
class QwenVQAModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        
        self.config = config
        self.embedder = Qwen3VLEmbedder(
            model_name_or_path=config.MODEL_NAME,
            max_length=100_000,
            max_frames=None,
            fps=None
        )
        
        # Applying your original video processor overrides
        vp = self.embedder.processor.video_processor
        vp.do_resize = False
        vp.max_frames = 100_000
        vp.max_pixels = config.MAX_PIXELS 
        vp.total_pixels = config.TOTAL_PIXELS

        self.embedder.model = self.embedder.model.to(dtype=config.DTYPE, device=config.DEVICE)
        
        lora_config = LoraConfig(
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=config.LORA_DROPOUT,
            bias="none",
            target_modules=config.LORA_TARGET_MODULES,
            task_type="FEATURE_EXTRACTION"
        )
        self.embedder.model.eval()
        self.model = get_peft_model(self.embedder.model, lora_config)
        
        hidden_dim = self.embedder.model.config.vision_config.out_hidden_size
        self.regressor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        ).to(config.DEVICE)

    def forward(self, processed):
        outputs = self.model.forward(**processed)
        last_hidden = outputs["last_hidden_state"]
        mask = outputs["attention_mask"].unsqueeze(-1)
        
        # Masked Global Average Pooling
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return self.regressor(summed / counts).squeeze()