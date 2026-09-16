"""TRIAD spherical watermark encoder, invariant decoder, and supporting modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.o3 import FromS2Grid, ToS2Grid, FullyConnectedTensorProduct

from .backbone import GatedBlock
from .unet_modules import UNet

# ============================================================================

# ============================================================================
class BispectrumDecoder(nn.Module):
    """Decode binary watermarks using third-order scalar contractions TP(TP(f, f), f)."""
    def __init__(
        self,
        latent_dim=8,
        resolution=256,
        lmax=16,
        embed_irreps="32x1e + 16x2e",
        hidden_irreps = "32x0e + 64x1e + 32x2e",
        middle_layers = 32,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.resolution = resolution
        self.embed_irreps = o3.Irreps(embed_irreps)
        self.embed_dim = self.embed_irreps.dim
        self.middle_layers = middle_layers
        # 1. Lifting
        self.lift = FromS2Grid(res=(resolution,2*resolution), lmax=lmax, normalization='component')
        sh_spec = o3.Irreps.spherical_harmonics(lmax)
        self.input_irreps = sh_spec * 3 
        invariant_dim = latent_dim *32
        
        self.adapter = o3.Linear(self.input_irreps, self.embed_irreps)
        
        # 3. Backbone
        self.backbone = nn.Sequential(
            GatedBlock(self.embed_irreps, self.embed_irreps),
            GatedBlock(self.embed_irreps, self.embed_irreps),
        )
        num_groups = 4
        self.num_groups = num_groups
        self.full_irreps = o3.Irreps(embed_irreps)
        
        
        
        assert latent_dim % num_groups == 0
        self.sub_latent_dim = latent_dim // num_groups
        
        
        sub_list = []
        for mul, ir in self.full_irreps:
            sub_list.append((mul // num_groups, ir))
        self.sub_irreps = o3.Irreps(sub_list)
        
        
        # self.embed_irreps= self.sub_irreps
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.hidden_scalars = o3.Irreps("32x0e ")
        self.hidden_others = o3.Irreps(" 32x14e +  64x6e + 64x8e +32x1e+32x2e  +16x3e + 16x4e+ 16x5e+16x7e ")
        self.hidden_others = o3.Irreps([
            (mul, ir) for mul, ir in self.hidden_irreps if ir.l > 0
        ])
        
        # self.tp_ff = FullyConnectedTensorProduct(
        #     self.embed_irreps,
        #     self.embed_irreps,
        #     self.hidden_irreps,
        # )
        self.tp_ff = FullyConnectedTensorProduct(
            self.sub_irreps,
            self.sub_irreps,
            self.hidden_irreps,
        )
        
        
        # self.tp_hf = FullyConnectedTensorProduct(
        #     self.hidden_irreps,
        #     self.embed_irreps,
        
        # )
        # self.tp_hf = FullyConnectedTensorProduct(
        #     self.hidden_others,
        #     self.embed_irreps,
        
        # )
        self.tp_hf = FullyConnectedTensorProduct(
            self.hidden_others,
            self.sub_irreps,
            f"{invariant_dim}x0e",  
        )
        # 5. Readout
        feature_dim = invariant_dim + self.hidden_scalars.dim
        self.readout = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 256),
            nn.SiLU(),
            # nn.Linear(512, 256),
            # nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, latent_dim//num_groups),
        )
        
    def forward(self, x_watermarked, return_features=False):
        B = x_watermarked.shape[0]
        
        # Lifting
        f_spec = self.lift(x_watermarked)
        f = f_spec.reshape(B, -1)
        f = self.adapter(f)
        f = self.backbone(f)  # [B, embed_dim]
        
        f_groups = self._safe_split(f)

        w_preds = []
        for i in range(self.num_groups):
            f_sub = f_groups[i]
            h = self.tp_ff(f_sub, f_sub)  # [B, hidden_dim]
            h_scalars = h[:, :self.hidden_scalars.dim]
            h = h[:, self.hidden_scalars.dim:]  
            
            features = self.tp_hf(h, f_sub)  # [B, latent_dim * 4]
            features = torch.cat([h_scalars, features], dim=-1)
            # Readout
            w_pred = self.readout(features)
            w_preds.append(w_pred)
            
        if return_features:
            return torch.cat(w_preds, dim=-1), features
        else:
            return torch.cat(w_preds, dim=-1)
        
        
        h = self.tp_ff(f, f)  # [B, hidden_dim]
        
        h_scalars = h[:, :self.hidden_scalars.dim]
        h = h[:, self.hidden_scalars.dim:]  
        
        features = self.tp_hf(h, f)  # [B, latent_dim * 4]
        
        features = torch.cat([h_scalars, features], dim=-1)
        # Readout
        w_pred = self.readout(features)  # [B, latent_dim]
        
        return w_pred
    def _safe_split(self, f):
        """Split multiplicities into groups using the encoder's irrep layout."""
        splits = [[] for _ in range(self.num_groups)]
        start_idx = 0
        for mul, ir in self.full_irreps:
            dim = ir.dim
            total_dim = mul * dim
            chunk = f[..., start_idx : start_idx + total_dim]
            start_idx += total_dim
            
            sub_dim = (mul // self.num_groups) * dim
            chunk_reshaped = chunk.view(f.shape[0], self.num_groups, sub_dim)
            for g in range(self.num_groups):
                splits[g].append(chunk_reshaped[:, g, :])
        
        return [torch.cat(s, dim=-1) for s in splits]

# ============================================================================

# ============================================================================
class BetterBispectrumEncoder(nn.Module):
    """Learn watermark-dependent spectral residuals with a content-dependent spatial mask."""
    def __init__(
        self,
        latent_dim=8,
        resolution=256,
        lmax=16,
        embed_irreps="32x1e + 16x2e",
        hidden_irreps = "32x0e + 64x1e + 32x2e",
        middle_layers = 32
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.resolution = resolution
        self.embed_irreps = o3.Irreps(embed_irreps)
        self.embed_dim = self.embed_irreps.dim
        
        
        middle_layers =3
        self.lift = FromS2Grid(res=(resolution, 2*resolution), lmax=lmax, normalization='component')
        sh_spec = o3.Irreps.spherical_harmonics(lmax)
        self.input_irreps = sh_spec * middle_layers
        
        self.adapter = o3.Linear(self.input_irreps, self.embed_irreps)
        
        self.backbone = nn.Sequential(
            GatedBlock(self.embed_irreps, self.embed_irreps),
            GatedBlock(self.embed_irreps, self.embed_irreps),
        )
        
        
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.hidden_dim = self.hidden_irreps.dim
        
        self.tp_ff = FullyConnectedTensorProduct(
            self.embed_irreps,
            self.embed_irreps,
            self.hidden_irreps,
        )
        self.w_embed_dim = 512  
        
        self.w_irreps = o3.Irreps(f"{self.w_embed_dim}x0e")
        # self.tp_injector = FullyConnectedTensorProduct(
        
        
        
        # )
        self.f_norm = nn.InstanceNorm1d(self.embed_dim, affine=False)
        
        self.scale_0 = nn.Parameter(torch.ones(latent_dim, self.hidden_dim))
        self.scale_1 = nn.Parameter(torch.ones(latent_dim, self.hidden_dim))
        self.shift_0 = nn.Parameter(torch.zeros(latent_dim, self.hidden_dim))
        self.shift_1 = nn.Parameter(torch.zeros(latent_dim, self.hidden_dim))
        self._init_modulation_params()
        num_groups = 4
        self.num_groups = num_groups
        
        self.w_encoder = nn.Sequential(
            nn.Linear(latent_dim//self.num_groups, 256),
            nn.SiLU(),
            nn.Linear(256, self.w_embed_dim), 
        )
        
        self.middle_layers = middle_layers
        
        self.output_adapter = o3.Linear(self.embed_irreps, self.input_irreps)
        self.to_s2 = ToS2Grid(lmax=lmax, res=(resolution,2*resolution), normalization='component')
        
        
        self.strength = nn.Parameter(torch.tensor(0.1))
        
        self.injector = RiemannianAdaptiveInjector(
            irreps_feature=self.embed_irreps, 
            resolution=resolution,
            lmax=lmax
        )
        self.mask_predictor = PerceptualMaskNet()
        # self.mask_predictor = ContextAwareMaskNet(in_channels=3)
        
        
        self.register_buffer('geo_prior', self._generate_geo_prior(resolution))
        self.jnd_module = JNDModule(edge_gain=10.0, base_visibility=0.01)
        self.pixel_mix = nn.Conv2d(3, self.middle_layers, kernel_size=1)
        # self.unet = UNetFusion(self.middle_layers,self.middle_layers * 2)
        self.unet = UNet(3+3,3)
        self.image_conv = nn.Conv2d(3, self.middle_layers, kernel_size=1)
        self.jnd  = JND(preprocess=denormalize, postprocess=normalize)
          
        self.full_irreps = o3.Irreps(embed_irreps)
        sub_irreps_list = []
        for mul, ir in self.full_irreps:
            assert mul % num_groups == 0, \
                f"Multiplicity {mul} must be divisible by {num_groups}; adjust the configuration"
            sub_irreps_list.append((mul // num_groups, ir))
        
        self.sub_embed_irreps = o3.Irreps(sub_irreps_list)
        self.tp_injector = FullyConnectedTensorProduct(
            self.sub_embed_irreps,    
            self.w_irreps,         
            self.sub_embed_irreps      
        )
        
        assert latent_dim % num_groups == 0
        self.sub_w_dim = latent_dim // num_groups
        self.embed_dim = self.sub_embed_irreps.dim
        self.combiner = nn.Sequential(
        
        nn.Linear(2*self.embed_dim, 2056), 
        nn.LayerNorm(2056),
        nn.SiLU(),
        nn.Linear(2056, self.embed_dim)
    )
        self.watermark_generator = nn.Sequential(
            nn.Linear(latent_dim//self.num_groups, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, self.embed_dim)
        )
        self.weighter = DynamicGroupWeighter(
            num_groups=self.num_groups,
            embed_irreps_per_group=self.sub_embed_irreps
        )
    def _generate_geo_prior(self, H, W=None):
        if W is None: W = 2 * H
        theta = torch.linspace(0, torch.pi, H).view(1, 1, H, 1)
        return torch.sin(theta) 
    def _init_modulation_params(self):
        with torch.no_grad():
            self.scale_0.data.fill_(1.0)
            self.scale_1.data.fill_(1.0)
            self.shift_0.data.normal_(0, 0.02)
            self.shift_1.data.normal_(0, 0.02)

    def compute_target_h(self, h_orig, w):
        """Construct target bispectrum features from bit-conditioned scale and shift parameters."""
        B = h_orig.shape[0]
        scale_total = torch.ones(B, self.hidden_dim, device=h_orig.device)
        shift_total = torch.zeros(B, self.hidden_dim, device=h_orig.device)
        
        for i in range(self.latent_dim):
            w_i = w[:, i:i+1]
            scale_i = w_i * self.scale_1[i:i+1] + (1 - w_i) * self.scale_0[i:i+1]
            shift_i = w_i * self.shift_1[i:i+1] + (1 - w_i) * self.shift_0[i:i+1]
            scale_total = scale_total * scale_i
            shift_total = shift_total + shift_i
            
        return h_orig * scale_total + shift_total

    def forward(self, x_s2, w, return_features=True):
        B = x_s2.shape[0]
        # x_mixed = self.pixel_mix(x_s2)
        x_mixed = x_s2
        
        f_spec = self.lift(x_mixed)
        f_flat = f_spec.reshape(B, -1)
        f = self.adapter(f_flat)
        f_orig = self.backbone(f)
        # f_unit = self.f_norm(f_orig).squeeze(2)
        
        h_orig = self.tp_ff(f_orig, f_orig)
        h_target = self.compute_target_h(h_orig, w)
        
        
        f_groups = self._safe_split(f_orig) 
        w_groups = torch.chunk(w, self.num_groups, dim=-1) 
        delta_groups = []
        for i in range(self.num_groups):
            delta_f = self.watermark_generator(w_groups[i])
            w_feat = self.w_encoder(w_groups[i])
            f_orig = self.tp_injector(f_groups[i], w_feat)
            d = self.combiner(torch.cat([f_orig, delta_f], dim=-1))
            delta_groups.append(d)
        weighted_deltas = self.weighter(f_groups, delta_groups)
        delta_total = self._safe_merge(weighted_deltas)
        
        strength = F.softplus(self.strength)
        # f_watermarked = f_orig + strength * delta_f
        # delta = self.injector(delta_f, x_s2)
        # f_watermarked = self.combiner(torch.cat([f_orig, delta_f], dim=-1))
        # f_watermarked = delta_f
        
        f_watermarked = delta_total
        h_actual = self.tp_ff(f_watermarked, f_watermarked)
        
        
        f_out = self.output_adapter(f_watermarked)
        f_out_folded = f_out.reshape(B, self.middle_layers, -1)
        residual = self.to_s2(f_out_folded)
        
        if residual.shape[-2:] != x_s2.shape[-2:]:
            residual = F.interpolate(residual, size=x_s2.shape[-2:], 
                                     mode='bilinear', align_corners=False)
        mask_content = self.mask_predictor(x_s2)
        # mask_content = self.jnd_module(x_s2)
        final_mask = mask_content * self.geo_prior   
          
        # x_out = F.tanh(x_s2 + (residual)* final_mask)
        x_out = x_s2 + (residual)* mask_content
        image_tensor = self.image_conv(x_s2)
        # x_out  = self.unet(image_tensor, residual)
        # x_out = self.unet(torch.cat([x_s2,residual],dim=1))
        if return_features:
            return x_out, h_actual, h_target, f_watermarked
        return x_out
    def _safe_split(self, f):
        """Split each irrep's multiplicity evenly, then concatenate corresponding slices per group."""
        splits = [[] for _ in range(self.num_groups)]
        start_idx = 0
        
        for mul, ir in self.full_irreps:
            dim = ir.dim
            total_dim = mul * dim
            
            
            chunk = f[..., start_idx : start_idx + total_dim]
            start_idx += total_dim
            
            
            # chunk: [B, mul * dim] -> reshape -> [B, groups, sub_mul * dim]
            sub_dim = (mul // self.num_groups) * dim
            chunk_reshaped = chunk.view(f.shape[0], self.num_groups, sub_dim)
            
            for g in range(self.num_groups):
                splits[g].append(chunk_reshaped[:, g, :])
        
        
        final_groups = [torch.cat(s, dim=-1) for s in splits]
        return final_groups

    def _safe_merge(self, delta_groups):
        """Reassemble groupwise residuals in the original e3nn irrep ordering."""
        
        
        merged_chunks = []
        
        
        start_idx = 0
        for mul, ir in self.sub_embed_irreps:
            dim = ir.dim
            sub_len = mul * dim
            
            
            type_chunks = []
            for g in range(self.num_groups):
                
                d = delta_groups[g]
                type_chunks.append(d[..., start_idx : start_idx + sub_len])
            
            
            merged_chunks.append(torch.cat(type_chunks, dim=-1))
            
            start_idx += sub_len 
            
        
        return torch.cat(merged_chunks, dim=-1)

# ============================================================================

# ============================================================================
class MatchedBispectrumSystem(nn.Module):
    """Combine the spectral watermark encoder and invariant bispectrum decoder."""
    def __init__(
        self,
        latent_dim=8,
        resolution=256,
        lmax=16,
        embed_irreps="32x1e + 16x2e",
        hidden_irreps = "32x0e + 64x1e + 32x2e",
        middle_layers = 32
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.resolution = resolution
        self.embed_irreps = o3.Irreps(embed_irreps)
        
        
        self.embedder = BetterBispectrumEncoder(
            latent_dim=latent_dim,
            resolution=resolution,
            lmax=lmax,
            embed_irreps=embed_irreps,
            hidden_irreps=hidden_irreps,
            middle_layers = middle_layers,
        )
        
        self.extractor = BispectrumDecoder(
            latent_dim=latent_dim,
            resolution=resolution,
            lmax=lmax,
            embed_irreps=embed_irreps,
            hidden_irreps=hidden_irreps,
            middle_layers = middle_layers,
        )
        
        
        # self.extractor.tp_ff = self.embedder.tp_ff
        
    def forward(self, x, w, attack_fn=None):
        """Embed w [B, latent_dim] in x [B, 3, H, W], optionally attack, and return image, logits and features."""
        x_wm, h_actual, h_target, f_wm = self.embedder(x, w, return_features=True)
        
        if attack_fn is not None:
            x_attacked = attack_fn(x_wm)
        else:
            x_attacked = x_wm
        
        w_pred = self.extractor(x_attacked)
        
        return {
            'x_wm': x_wm,
            'x_watermarked': x_wm,
            'x_attacked': x_attacked,
            'w_pred': w_pred,
            'f_watermarked': f_wm,
            'h_actual': h_actual,
            'h_target': h_target,
        }
    
    def embed(self, x, w):
        """Return the watermarked ERP tensor."""
        return self.embedder(x, w, return_features=False)
    
    def extract(self, x_wm):
        """Return watermark logits from an ERP tensor."""
        return self.extractor(x_wm)
    
    def get_signatures(self):
        """Return the encoder's scale and shift parameters."""
        return {
            'scale_0': self.embedder.scale_0,
            'scale_1': self.embedder.scale_1,
            'shift_0': self.embedder.shift_0,
            'shift_1': self.embedder.shift_1,
        }
    
    def compute_orthogonality_loss(self):
        """Compute a Gram-matrix orthogonality penalty on modulation parameters."""
        diff_scale = self.embedder.scale_1 - self.embedder.scale_0
        gram = torch.matmul(diff_scale, diff_scale.T)
        I = torch.eye(self.latent_dim, device=gram.device)
        loss_scale = ((gram / gram.diag().mean().clamp(min=1e-6) - I) ** 2).mean()
        
        diff_shift = self.embedder.shift_1 - self.embedder.shift_0
        gram_shift = torch.matmul(diff_shift, diff_shift.T)
        loss_shift = ((gram_shift / gram_shift.diag().mean().clamp(min=1e-6) - I) ** 2).mean()
        
        return loss_scale + loss_shift
    
    def get_num_parameters(self):
        """Count model parameters."""
        return sum(p.numel() for p in self.parameters())
class SparseSpectralTransform(nn.Module):
    """Scatter sparse irrep coefficients into a dense SH spectrum for grid transforms, and gather on inversion."""
    def __init__(self, irreps_str, resolution=256, lmax=None):
        super().__init__()
        self.irreps = o3.Irreps(irreps_str)
        
        
        if lmax is None:
            lmax = self.irreps.lmax
        self.lmax = lmax
        self.dim_dense = (lmax + 1) ** 2
        
        
        self._to_grid = ToS2Grid(lmax=lmax, res=resolution, normalization='component')
        self._from_grid = FromS2Grid(lmax=lmax, res=resolution, normalization='component')
        
        
        
        self.slices_in = []  
        self.indices_dense = [] 
        self.shapes_spatial = [] 
        
        start_idx = 0
        for (mul, ir) in self.irreps:
            dim_chunk = mul * (2 * ir.l + 1)
            
            
            self.slices_in.append(slice(start_idx, start_idx + dim_chunk))
            
            
            
            # l=0: [0]
            # l=1: [1, 2, 3]
            # l=2: [4, 5, 6, 7, 8] ...
            
            dense_start = ir.l ** 2
            dense_end = (ir.l + 1) ** 2
            
            
            
            self.indices_dense.append((dense_start, dense_end))
            self.shapes_spatial.append(mul)
            
            start_idx += dim_chunk

    def to_spatial(self, x_spec):
        """Map [B, total_dim] spectral features to [B, sum_of_multiplicities, H, W] spatial features."""
        B = x_spec.shape[0]
        outputs = []
        
        for i, (mul, ir) in enumerate(self.irreps):
            
            chunk = x_spec[:, self.slices_in[i]]
            
            
            chunk = chunk.reshape(B, mul, -1)
            
            
            
            dense_spec = torch.zeros(B, mul, self.dim_dense, device=x_spec.device)
            
            
            l_start, l_end = self.indices_dense[i]
            dense_spec[:, :, l_start:l_end] = chunk
            
            
            
            grid = self._to_grid(dense_spec) 
            outputs.append(grid)
            
        
        return torch.cat(outputs, dim=1)

    def from_spatial(self, x_grid):
        """
        Args:
            x_grid: [B, total_channels, H, W]
        Returns:
            x_spec: [B, total_dim]
        """
        B = x_grid.shape[0]
        outputs = []
        start_ch = 0
        
        for i, (mul, ir) in enumerate(self.irreps):
            
            grid_chunk = x_grid[:, start_ch : start_ch + mul, :, :]
            start_ch += mul
            
            
            dense_spec = self._from_grid(grid_chunk)
            
            
            l_start, l_end = self.indices_dense[i]
            spec_chunk = dense_spec[:, :, l_start:l_end]
            
            
            outputs.append(spec_chunk.reshape(B, -1))
            
        return torch.cat(outputs, dim=1)
    
class RiemannianAdaptiveInjector(nn.Module):
    def __init__(self, irreps_feature, resolution=256, lmax=16):
        super().__init__()
        
        
        self.transformer = SparseSpectralTransform(
            irreps_str=irreps_feature,
            resolution=resolution,
            lmax=lmax
        )
        
        
        self.dim_total = self.transformer.irreps.dim
        self.channel_attention = nn.Sequential(
            nn.Linear(self.dim_total, self.dim_total),
            nn.Sigmoid()
        )
        
        self.diffusion_t = nn.Parameter(torch.tensor(0.01))

    
    def _compute_saliency(self, x_img):
        """Compute image saliency from local gradient magnitude."""
        B, C, H, W = x_img.shape
        dx = x_img[..., 1:, :] - x_img[..., :-1, :]
        dy = x_img[..., :, 1:] - x_img[..., :, :-1]
        dx = F.pad(dx, (0, 0, 0, 1))
        dy = F.pad(dy, (0, 1, 0, 0))
        grad_mag = torch.sqrt(dx**2 + dy**2 + 1e-8).mean(dim=1, keepdim=True)
        return grad_mag

    def _get_riemannian_weights(self, B, H_grid, W_grid, device):
        """Return the spherical area weighting sin(theta)."""
        theta = torch.linspace(0, torch.pi, H_grid, device=device).view(1, 1, H_grid, 1).expand(B, 1, H_grid, W_grid)
        return torch.sin(theta)

    def _spectral_diffusion(self, f_spec):
        """Manifold Diffusion"""
        diffused_coeffs = []
        start_idx = 0
        t = F.softplus(self.diffusion_t)
        
        
        for (mul, ir) in self.transformer.irreps:
            dim = mul * (2 * ir.l + 1)
            decay = torch.exp(-t * ir.l * (ir.l + 1))
            chunk = f_spec[:, start_idx : start_idx + dim]
            diffused_coeffs.append(chunk * decay)
            start_idx += dim
            
        return torch.cat(diffused_coeffs, dim=1)

    def forward(self, delta_f_spec, x_img):
        B = x_img.shape[0]
        
        # 1. Sparse Spectral -> Spatial Grid
        # delta_grid: [B, 96, H_g, W_g] (96 = 64+32)
        delta_grid = self.transformer.to_spatial(delta_f_spec)
        H_grid, W_grid = delta_grid.shape[-2:]
        
        
        saliency_img = self._compute_saliency(x_img)
        if (H_grid, W_grid) != saliency_img.shape[-2:]:
            saliency_grid = F.interpolate(saliency_img, size=(H_grid, W_grid), mode='bilinear', align_corners=False)
        else:
            saliency_grid = saliency_img
            
        geo_weight = self._get_riemannian_weights(B, H_grid, W_grid, x_img.device)
        attention_grid = saliency_grid * geo_weight
        attention_grid = 2.0 * (attention_grid / (attention_grid.mean() + 1e-6))
        
        
        
        delta_grid_modulated = delta_grid * attention_grid
        
        # 4. Spatial Grid -> Sparse Spectral
        delta_spec_modulated = self.transformer.from_spatial(delta_grid_modulated)
        
        
        delta_final = self._spectral_diffusion(delta_spec_modulated)
        
        
        gate = self.channel_attention(delta_final)
        delta_final = delta_final * gate
        
        return delta_final
class PerceptualMaskNet(nn.Module):
    """Predict a single-channel gain map in [0, 1] from the original RGB image."""
    def __init__(self, in_channels=3):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 1, kernel_size=1), 
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.net(x)
    

class JNDModule(nn.Module):
    def __init__(self, kernel_size=3, edge_gain=5.0, base_visibility=0.01):
        super().__init__()
        self.edge_gain = edge_gain
        self.base_visibility = base_visibility 
        
        
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.padding = 1

    def forward(self, x):
        """Map RGB images [B, 3, H, W] to visibility masks [B, 1, H, W]."""
        
        
        if x.min() < 0:
            x_norm = (x + 1) / 2
        else:
            x_norm = x
            
        gray = 0.299 * x_norm[:, 0:1] + 0.587 * x_norm[:, 1:2] + 0.114 * x_norm[:, 2:3]
        
        
        grad_x = F.conv2d(gray, self.sobel_x, padding=self.padding)
        grad_y = F.conv2d(gray, self.sobel_y, padding=self.padding)
        magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        
        
        
        
        
        
        
        jnd_map = torch.sigmoid(magnitude * self.edge_gain)
        
        
        
        
        
        
        jnd_map = self.base_visibility + (1 - self.base_visibility) * jnd_map
        
        return jnd_map
    
class JND(nn.Module):
    """ https://ieeexplore.ieee.org/document/7885108 """

    def __init__(
        self,
        preprocess=lambda x: x,
        postprocess=lambda x: x,
        in_channels=1,
        out_channels=3,
    ) -> None:
        super(JND, self).__init__()

        # setup input and output methods
        self.in_channels = in_channels
        self.out_channels = out_channels
        groups = self.in_channels

        # create kernels
        kernel_x = torch.tensor(
            [[-1., 0., 1.],
             [-2., 0., 2.],
             [-1., 0., 1.]]
        ).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.tensor(
            [[1., 2., 1.],
             [0., 0., 0.],
             [-1., -2., -1.]]
        ).unsqueeze(0).unsqueeze(0)
        kernel_lum = torch.tensor(
            [[1., 1., 1., 1., 1.],
             [1., 2., 2., 2., 1.],
             [1., 2., 0., 2., 1.],
             [1., 2., 2., 2., 1.],
             [1., 1., 1., 1., 1.]]
        ).unsqueeze(0).unsqueeze(0)

        # Expand kernels for 3 input channels and 3 output channels, apply the same filter to each channel
        kernel_x = kernel_x.repeat(groups, 1, 1, 1)
        kernel_y = kernel_y.repeat(groups, 1, 1, 1)
        kernel_lum = kernel_lum.repeat(groups, 1, 1, 1)

        self.conv_x = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1, bias=False, groups=groups)
        self.conv_y = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1, bias=False, groups=groups)
        self.conv_lum = nn.Conv2d(3, 3, kernel_size=(5, 5), padding=2, bias=False, groups=groups)

        self.conv_x.weight = nn.Parameter(kernel_x, requires_grad=False)
        self.conv_y.weight = nn.Parameter(kernel_y, requires_grad=False)
        self.conv_lum.weight = nn.Parameter(kernel_lum, requires_grad=False)

        # setup pre and post processing
        self.preprocess = preprocess
        self.postprocess = postprocess

    def jnd_la(self, x, alpha=1.0, eps=1e-5):
        """ Luminance masking: x must be in [0,255] """
        la = self.conv_lum(x) / 32
        mask_lum = la <= 127
        la[mask_lum] = 17 * (1 - torch.sqrt(la[mask_lum] / 127 + eps))
        la[~mask_lum] = 3 / 128 * (la[~mask_lum] - 127) + 3
        return alpha * la

    def jnd_cm(self, x, beta=0.117, eps=1e-5):
        """ Contrast masking: x must be in [0,255] """
        grad_x = self.conv_x(x)
        grad_y = self.conv_y(x)
        cm = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        cm = 16 * cm ** 2.4 / (cm ** 2 + 26 ** 2)
        return beta * cm

    # @torch.no_grad()
    def heatmaps(
        self,
        imgs: torch.Tensor,
        clc: float = 0.3
    ) -> torch.Tensor:
        """ imgs must be in [0,1] after preprocess """
        imgs = 255 * imgs
        rgbs = torch.tensor([0.299, 0.587, 0.114])
        if self.in_channels == 1:
            imgs = rgbs[0] * imgs[..., 0:1, :, :] + rgbs[1] * imgs[..., 1:2, :, :] + rgbs[2] * imgs[..., 2:3, :, :]  # luminance: b 1 h w
        la = self.jnd_la(imgs)
        cm = self.jnd_cm(imgs)
        hmaps = torch.clamp_min(la + cm - clc * torch.minimum(la, cm), 0)  # b 1 or 3 h w
        if self.out_channels == 3 and self.in_channels == 1:
            hmaps = hmaps.repeat(1, 3, 1, 1)  # b 3 h w
        elif self.out_channels == 1 and self.in_channels == 3:
            hmaps = torch.sum(hmaps / 3, dim=1, keepdim=True)  # b 1 h w
        return hmaps / 255

    def forward(self, imgs: torch.Tensor, imgs_w: torch.Tensor, alpha: float = 1.0, blue: bool = False) -> torch.Tensor:
        """ imgs and deltas must be in [0,1] after preprocess """
        imgs = self.preprocess(imgs)
        imgs_w = self.preprocess(imgs_w)
        hmaps = self.heatmaps(imgs, clc=0.3)
        if blue:
            hmaps[:, 0] = hmaps[:, 0] * 0.75
            hmaps[:, 1] = hmaps[:, 1] * 0.50
            hmaps[:, 2] = hmaps[:, 2] * 1.00
        imgs_w = imgs + alpha * hmaps * (imgs_w - imgs)
        return self.postprocess(imgs_w)
    
def normalize(images):
    """
    Normalize an image array to [-1,1].
    """
    return ((images - 0.5) * 2).clamp(-1, 1)


def denormalize(images):
    """
    Denormalize an image array to [0,1].
    """
    return (images / 2 + 0.5).clamp(0, 1)
class CoordinateInjection(nn.Module):
    """Append normalized latitude and longitude as spatial feature channels."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        
        
        theta = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        
        
        phi = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        
        
        return torch.cat([x, theta, phi], dim=1)

class DilatedBlock(nn.Module):
    """Expand the receptive field with a resolution-preserving dilated residual block."""
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            
            
            nn.Conv2d(channels, channels, kernel_size=3, 
                      padding=dilation, dilation=dilation, padding_mode='circular'),
            nn.GroupNorm(8, channels),
            nn.SiLU() 
        )
        
    def forward(self, x):
        return x + self.block(x) 

class ContextAwareMaskNet(nn.Module):
    """
    SOTA Design: Coordinate-Aware + Wide Receptive Field
    """
    def __init__(self, in_channels=3):
        super().__init__()
        
        self.coord_add = CoordinateInjection()
        
        
        self.entry = nn.Sequential(
            nn.Conv2d(in_channels + 2, 32, kernel_size=3, padding=1, padding_mode='circular'),
            nn.SiLU()
        )
        
        
        
        # Layer 1 (d=1): 3x3
        # Layer 2 (d=2): +4 -> 7x7
        # Layer 3 (d=4): +8 -> 15x15
        # Layer 4 (d=8): +16 -> 31x31
        # Layer 5 (d=16): +32 -> 63x63
        self.body = nn.Sequential(
            DilatedBlock(32, dilation=1),
            DilatedBlock(32, dilation=2),
            DilatedBlock(32, dilation=4),
            DilatedBlock(32, dilation=8) 
        )
        
        
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 32),
            nn.Sigmoid()
        )
        
        
        self.exit = nn.Conv2d(32, 1, kernel_size=1)
        
    def forward(self, x):
        
        x_in = self.coord_add(x)
        
        
        feat = self.entry(x_in)
        
        
        feat = self.body(feat)
        
        
        # global_scale: [B, 32, 1, 1]
        b, c, _, _ = feat.shape
        global_stat = self.global_pool(feat).view(b, c)
        global_scale = self.global_fc(global_stat).view(b, c, 1, 1)
        feat = feat * global_scale
        
        
        mask = torch.sigmoid(self.exit(feat))
        
        return mask
class DynamicGroupWeighter(nn.Module):
    def __init__(self, num_groups, embed_irreps_per_group):
        super().__init__()
        self.num_groups = num_groups
        
        self.sub_irreps = o3.Irreps(embed_irreps_per_group)
        
        
        
        self.norm = o3.Norm(self.sub_irreps)
        
        
        
        
        
        self.mlp = nn.Sequential(
            nn.Linear(num_groups, num_groups * 2), 
            nn.LayerNorm(num_groups * 2),
            nn.SiLU(),
            nn.Linear(num_groups * 2, num_groups),
            nn.Sigmoid() 
        )
        
        
        self.scale = nn.Parameter(torch.ones(num_groups) * 2.0)

    def forward(self, f_groups, delta_groups):
        """Weight delta_groups using the energies of the corresponding original f_groups."""
        B = delta_groups[0].shape[0]
        
        
        
        group_energies = []
        for f_g in f_groups:
            # f_g: [B, sub_dim] -> norm -> [B, 1]
            n = self.norm(f_g) 
            
            group_energies.append(n.mean(dim=-1, keepdim=True))
            
        # [B, num_groups]
        global_desc = torch.cat(group_energies, dim=-1)
        
        
        # weights: [B, num_groups]
        weights = self.mlp(global_desc) * self.scale
        
        
        weighted_deltas = []
        for i in range(self.num_groups):
            # w: [B, 1]
            w = weights[:, i:i+1]
            
            weighted_deltas.append(delta_groups[i] * w)
            
        return weighted_deltas
