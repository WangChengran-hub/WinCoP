from __future__ import annotations

import numpy as np
import torch

from gealign_ov.models.geoalign_ov_core import (
    GeoAlignOVCore,
    GeoAlignOVCoreOutput,
    overlap_average_window_logits,
    SLIDE_CROP,
    SLIDE_STRIDE,
)
from gealign_ov.data.sliding_windows import compute_sliding_windows
from gealign_ov.methods.relational_transport import (
    TRANSPORT_ALPHA,
    build_dino_relation_operator,
    coarse_query_logits_from_window,
    residual_to_window,
)


class GeoAlignAP4Core(GeoAlignOVCore):
    """
    Training-free AP4-only inference.

    A: frozen DINO top-K relation operator.
    P: global-coordinate consensus projection.

    X_0 = S
    X_{t+1} = A P X_t, t = 0,...,3

    No eigendecomposition, CSCT, QST, DP-QST or DC-QT.
    """

    @torch.inference_mode()
    def forward(self, normalized_rgb):
        self._validate_input(normalized_rgb)

        height, width = map(
            int, normalized_rgb.shape[-2:]
        )

        windows = compute_sliding_windows(
            image_height=height,
            image_width=width,
            crop_height=SLIDE_CROP,
            crop_width=SLIDE_CROP,
            stride_height=SLIDE_STRIDE,
            stride_width=SLIDE_STRIDE,
        )

        rgb_01 = self.input_adapter(normalized_rgb)
        seg_input = normalized_rgb.half()

        logits_list = []
        token_list = []
        graph_list = []
        signal_list = []
        coordinates = []

        for window in windows:
            ys, xs = window.slices

            logits = self.segearth_model.forward_feature(
                seg_input[:, :, ys, xs]
            )[0].float()

            tokens = self.dino_backbone(
                rgb_01[:, :, ys, xs]
            ).patch_tokens[0].float()

            if (
                logits.shape
                != (self.query_count, 224, 224)
                or tokens.shape != (196, 768)
            ):
                raise RuntimeError(
                    "AP4_FROZEN_WINDOW_INTERFACE_MISMATCH"
                )

            graph = build_dino_relation_operator(tokens)
            signal = coarse_query_logits_from_window(logits)

            logits_list.append(logits)
            token_list.append(tokens)
            graph_list.append(graph)
            signal_list.append(signal)

            y, x = int(ys.start), int(xs.start)

            for r in range(14):
                for c in range(14):
                    coordinates.append((
                        y + r * 16 + 8,
                        x + c * 16 + 8,
                    ))

        logits = torch.stack(logits_list)
        tokens = torch.stack(token_list)
        A = torch.stack(graph_list)
        S = torch.stack(signal_list)

        coords = np.asarray(coordinates, dtype=np.int64)

        _, inverse_np, counts_np = np.unique(
            coords,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )

        inverse = torch.as_tensor(
            inverse_np.copy(),
            dtype=torch.long,
            device=S.device,
        )

        counts = torch.as_tensor(
            counts_np.copy(),
            dtype=S.dtype,
            device=S.device,
        )

        def project(X):
            channels = X.shape[-1]

            pooled = torch.zeros(
                (len(counts_np), channels),
                device=X.device,
                dtype=X.dtype,
            )

            pooled.index_add_(
                0,
                inverse,
                X.reshape(-1, channels),
            )

            pooled /= counts[:, None]

            return pooled[inverse].reshape_as(X)

        state = S

        for _ in range(4):
            state = torch.bmm(A, project(state))

        refined = torch.stack([
            logits[wi]
            + float(TRANSPORT_ALPHA)
            * residual_to_window(state[wi] - S[wi])
            for wi in range(len(windows))
        ])

        base_canvas = overlap_average_window_logits(
            logits,
            windows,
            height,
            width,
        )

        refined_canvas = overlap_average_window_logits(
            refined,
            windows,
            height,
            width,
        )

        return GeoAlignOVCoreOutput(
            windows=windows,
            base_window_logits=logits,
            patch_tokens=tokens,
            refined_window_logits=refined,
            base_logits_canvas=base_canvas,
            refined_logits_canvas=refined_canvas,
        )
