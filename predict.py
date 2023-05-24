# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

import os
import tempfile
from copy import deepcopy
from pathlib import Path

import cog
import numpy as np
import torch
from PIL import Image
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm

from e4e_projection import projection as e4e_projection
from model import Discriminator, Generator
from util import align_face


class Predictor(cog.Predictor):
    def setup(self):
        pass

    @cog.input("input_face", type=Path, help="Photo of human face")
    @cog.input(
        "pretrained",
        type=str,
        default=None,
        help="Identifier of pretrained style",
        options=[
            "art",
            "arcane_multi",
            "sketch_multi",
            "arcane_jinx",
            "arcane_caitlyn",
            "jojo_yasuho",
            "jojo",
            "disney",
        ],
    )
    @cog.input("style_img_0", default=None, type=Path, help="Face style image (unused if pretrained style is set)")
    @cog.input("style_img_1", default=None, type=Path, help="Face style image (optional)")
    @cog.input("style_img_2", default=None, type=Path, help="Face style image (optional)")
    @cog.input("style_img_3", default=None, type=Path, help="Face style image (optional)")
    @cog.input(
        "preserve_color",
        default=False,
        type=bool,
        help="Preserve the colors of the original image",
    )
    @cog.input(
        "num_iter", default=200, type=int, min=0, help="Number of finetuning steps (unused if pretrained style is set)"
    )
    @cog.input(
        "alpha", default=1, type=float, min=0, max=1, help="Strength of finetuned style"
    )
    def predict(
        self,
        input_face,
        pretrained,
        style_img_0,
        style_img_1,
        style_img_2,
        style_img_3,
        preserve_color,
        num_iter,
        alpha,
        lambda_cycle: float=1.0
    ):

        device = "cuda"  # 'cuda' or 'cpu'

        latent_dim = 512

        # Load original generator
        original_generator = Generator(1024, latent_dim, 8, 2).to(device)
        ckpt = torch.load(
            "models/stylegan2-ffhq-config-f.pt",
            map_location=lambda storage, loc: storage,
        )
        original_generator.load_state_dict(ckpt["g_ema"], strict=False)

        # to be finetuned generators
        cartoon_generator = deepcopy(original_generator)
        human_generator = deepcopy(original_generator)


        transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # aligns and crops face
        aligned_face = align_face(str(input_face))
        aligned_cartoon_face = align_face(str(style_img_0))

        my_w = e4e_projection(aligned_face, "input_face.pt", device).unsqueeze(0)
        cartoon_w = e4e_projection(aligned_cartoon_face, "cartoon_face.pt", device).unsqueeze(0)
    
        if pretrained is not None:
            if (
                preserve_color
                and not (pretrained == "art")
                and not (pretrained == "sketch_multi")
            ):
                ckpt = f"{pretrained}_preserve_color.pt"
            else:
                ckpt = f"{pretrained}.pt"

            ckpt = torch.load(
                os.path.join("models", ckpt), map_location=lambda storage, loc: storage
            )
            cartoon_generator.load_state_dict(ckpt["g"], strict=False)

            with torch.no_grad():
                cartoon_generator.eval()
                stylized_face = cartoon_generator(my_w, input_is_latent=True)

        else:
            # finetune with new style images
            cartoon_targets = []
            cartoon_latents = []

            cartoon_imgs = [style_img_0, style_img_1, style_img_2, style_img_3]

            # Remove None values
            cartoon_imgs = [i for i in cartoon_imgs if i]

            for ind, cartoon_img in enumerate(cartoon_imgs):

                # crop and align the face
                cartoon_style_aligned = align_face(str(cartoon_img))

                out_path = f"cartoon_style_aligned_{ind}.jpg"
                cartoon_style_aligned.save(str(out_path))

                # GAN invert
                latent = e4e_projection(cartoon_style_aligned, f"style_img_{ind}.pt", device)

                cartoon_targets.append(transform(cartoon_style_aligned).to(device))
                cartoon_latents.append(latent.to(device))

            cartoon_targets = torch.stack(cartoon_targets, 0)
            cartoon_latents = torch.stack(cartoon_latents, 0) # latents are from human domain

            # Do the same operations above for second generator
            human_imgs = [input_face]
            human_targets = []
            human_latents = []
            for ind, human_img in enumerate(human_imgs):

                # crop and align the face
                human_style_aligned = align_face(str(human_img))

                out_path = f"human_style_aligned_{ind}.jpg"
                human_style_aligned.save(str(out_path))

                # GAN invert
                latent = e4e_projection(human_style_aligned, f"style_img_{ind}.pt", device)

                human_targets.append(transform(human_style_aligned).to(device))
                human_latents.append(latent.to(device))

            human_targets = torch.stack(human_targets, 0)
            human_latents = torch.stack(human_latents, 0)


            alpha = 1 - alpha
            # load discriminator for perceptual loss
            cartoon_discriminator = Discriminator(1024, 2).eval().to(device)
            human_discriminator = Discriminator(1024, 2).eval().to(device)

            ckpt = torch.load(
                "models/stylegan2-ffhq-config-f.pt",
                map_location=lambda storage, loc: storage,
            )
            cartoon_discriminator.load_state_dict(ckpt["d"], strict=False)

            cartoon_g_optim = optim.Adam(cartoon_generator.parameters(), lr=2e-3, betas=(0, 0.99))
            human_g_optim = optim.Adam(human_generator.parameters(), lr=2e-3, betas=(0, 0.99))

            # Which layers to swap for generating a family of plausible real images -> fake image
            if preserve_color:
                id_swap = [9, 11, 15, 16, 17]
            else:
                id_swap = list(range(7, cartoon_generator.n_latent))

            for idx in tqdm(range(num_iter)):
                cartoon_mean_w = (
                    cartoon_generator.get_latent(
                        torch.randn([cartoon_latents.size(0), latent_dim]).to(device)
                    )
                    .unsqueeze(1)
                    .repeat(1, cartoon_generator.n_latent, 1)
                )
                in_cartoon_latent = cartoon_latents.clone()
                in_cartoon_latent[:, id_swap] = (
                    alpha * cartoon_latents[:, id_swap] + (1 - alpha) * cartoon_mean_w[:, id_swap]
                )

                human_mean_w = (
                    human_generator.get_latent(
                        torch.randn([human_latents.size(0), latent_dim]).to(device)
                    )
                    .unsqueeze(1)
                    .repeat(1, human_generator.n_latent, 1)
                )
                in_human_latent = human_latents.clone()
                in_human_latent[:, id_swap] = (
                    alpha * human_latents[:, id_swap] + (1 - alpha) * human_mean_w[:, id_swap]
                )

                fake_cartoon_img = cartoon_generator(in_cartoon_latent, input_is_latent=True)
                fake_human_img = human_generator(in_human_latent, input_is_latent=True)

                with torch.no_grad():
                    real_cartoon_feat = cartoon_discriminator(cartoon_targets)
                    real_human_feat = human_discriminator(human_targets)
                fake_cartoon_feat = cartoon_discriminator(fake_cartoon_img)
                fake_human_feat = human_discriminator(fake_human_img)

                cartoon_loss = sum(
                    [F.l1_loss(a, b) for a, b in zip(fake_cartoon_feat, real_cartoon_feat)]
                ) / len(fake_cartoon_feat)
                human_loss = sum(
                    [F.l1_loss(a, b) for a, b in zip(fake_human_feat, real_human_feat)]
                ) / len(fake_human_feat)

                # Cycle loss
                cycle_cartoon = cartoon_generator(fake_human_img)
                cycle_human = human_generator(fake_cartoon_img)
                cycle_cartoon_loss = F.l1_loss(cartoon_targets, cycle_cartoon)
                cycle_human_loss = F.l1_loss(human_targets, cycle_human)

                cartoon_loss += lambda_cycle * cycle_cartoon_loss
                human_loss += lambda_cycle * cycle_human_loss

                cartoon_g_optim.zero_grad()
                human_g_optim.zero_grad()
                cartoon_loss.backward()
                human_loss.backward()
                cartoon_g_optim.step()
                human_g_optim.step()

            with torch.no_grad():
                cartoon_generator.eval()
                stylized_face = cartoon_generator(my_w, input_is_latent=True)

        stylized_face = stylized_face.cpu()
        np.save("stylized_face.npy", stylized_face)

        stylized_face = 1 + stylized_face
        stylized_face /= 2

        stylized_face = stylized_face[0]
        stylized_face = 255 * torch.clip(stylized_face, min=0, max=1)
        stylized_face = stylized_face.byte()

        stylized_face = stylized_face.permute(1, 2, 0).detach().numpy()
        stylized_face = Image.fromarray(stylized_face, mode="RGB")
        out_path = Path(tempfile.mkdtemp()) / "out.jpg"
        stylized_face.save(str(out_path))

        return out_path
