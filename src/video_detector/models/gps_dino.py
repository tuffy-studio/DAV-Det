import torch
import torch.nn as nn
from .classifier_modules import FlexibleMLP, Segment_Classifier_Reducer, Patch_Classifier_Reducer, vit_patch_clustering
from .dinov3 import DINOv3Model_LORA


class GPS_DINO(nn.Module):
    def __init__(self, backbone_name, layer_indices, use_lora=True, lora_r=32, lora_alpha=16, lora_dropout=0.1, use_deep_supervision=False, unfreeze_norm=True):
        super(GPS_DINO, self).__init__()

        self.use_deep_supervision = use_deep_supervision
        if self.use_deep_supervision:
            self.layer_indices = layer_indices
        else:
            self.layer_indices = [24]

        self.dinov3 = DINOv3Model_LORA(
            backbone_name=backbone_name,
            layer_indices=self.layer_indices,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            unfreeze_norm=unfreeze_norm
            )


        # main classifier 
        self.main_classifier = FlexibleMLP(input_size=1024*3, hidden_sizes=[1024*2, 1024*1], num_classes=1, drop_rates=[0.1, 0.2])

        # global classifier 输入是 cls token 
        self.global_classifier = FlexibleMLP(input_size=1024, hidden_sizes=[256], num_classes=1, drop_rates=[0.1])
        if self.use_deep_supervision:
            self.global_classifier_21 = FlexibleMLP(input_size=1024, hidden_sizes=[256], num_classes=1, drop_rates=[0.1])
            self.global_classifier_22 = FlexibleMLP(input_size=1024, hidden_sizes=[256], num_classes=1, drop_rates=[0.1])
            self.global_classifier_23 = FlexibleMLP(input_size=1024, hidden_sizes=[256], num_classes=1, drop_rates=[0.1])

        # patch classifier 输入是经过 adaptive aggregation 的 patch token
        self.patch_classifier = FlexibleMLP(input_size=1024, hidden_sizes=[256], num_classes=1, drop_rates=[0.1])
        if self.use_deep_supervision:
            self.patch_classifier_21 = FlexibleMLP(input_size=1024, hidden_sizes=[256], num_classes=1, drop_rates=[0.1])
            self.patch_classifier_22 = FlexibleMLP(input_size=1024, hidden_sizes=[256], num_classes=1, drop_rates=[0.1])
            self.patch_classifier_23 = FlexibleMLP(input_size=1024, hidden_sizes=[256], num_classes=1, drop_rates=[0.1])

        # segment classifier 输入是 Cluster Prototype tokens
        self.segment_classifier = FlexibleMLP(input_size=1024, hidden_sizes=[256], num_classes=1, drop_rates=[0.1])
        if self.use_deep_supervision:
            self.segment_classifier_21 = FlexibleMLP(input_size=1024, hidden_sizes=[256], num_classes=1, drop_rates=[0.1])
            self.segment_classifier_22 = FlexibleMLP(input_size=1024, hidden_sizes=[256], num_classes=1, drop_rates=[0.1])
            self.segment_classifier_23 = FlexibleMLP(input_size=1024, hidden_sizes=[256], num_classes=1, drop_rates=[0.1])

        # patch classifier & reducer
        self.patch_classifier_reducer = Patch_Classifier_Reducer(input_dim=1024, hidden_dim=256, temperature=0.07, topk_ratio=0.05)
        if self.use_deep_supervision:
            self.patch_classifier_reducer_21 = Patch_Classifier_Reducer(input_dim=1024, hidden_dim=256, temperature=0.07, topk_ratio=0.05)
            self.patch_classifier_reducer_22 = Patch_Classifier_Reducer(input_dim=1024, hidden_dim=256, temperature=0.07, topk_ratio=0.05)
            self.patch_classifier_reducer_23 = Patch_Classifier_Reducer(input_dim=1024, hidden_dim=256, temperature=0.07, topk_ratio=0.05)

        # segment clustering & classifier
        self.segment_classifier_reducer = Segment_Classifier_Reducer(input_dim=1024, hidden_dim=256, temperature=0.07, topk_ratio=0.1)
        if self.use_deep_supervision:
            self.segment_classifier_reducer_21 = Segment_Classifier_Reducer(input_dim=1024, hidden_dim=256, temperature=0.07, topk_ratio=0.1)
            self.segment_classifier_reducer_22 = Segment_Classifier_Reducer(input_dim=1024, hidden_dim=256, temperature=0.07, topk_ratio=0.1)
            self.segment_classifier_reducer_23 = Segment_Classifier_Reducer(input_dim=1024, hidden_dim=256, temperature=0.07, topk_ratio=0.1)

        # norm before classifier
        self.norm_global = nn.LayerNorm(1024)
        self.norm_patch = nn.LayerNorm(1024)
        self.norm_segment = nn.LayerNorm(1024)
        if self.use_deep_supervision:
            self.norm_global_21 = nn.LayerNorm(1024)
            self.norm_global_22 = nn.LayerNorm(1024)
            self.norm_global_23 = nn.LayerNorm(1024)

            self.norm_patch_21 = nn.LayerNorm(1024)
            self.norm_patch_22 = nn.LayerNorm(1024)
            self.norm_patch_23 = nn.LayerNorm(1024)

            self.norm_segment_21 = nn.LayerNorm(1024)
            self.norm_segment_22 = nn.LayerNorm(1024)
            self.norm_segment_23 = nn.LayerNorm(1024)

    def layerwise_segment(self, patch_tokens, cluster_labels):

        K = cluster_labels.max().item() + 1
        protos = []

        for k in range(K):
            idx = (cluster_labels == k)
            if idx.sum() > 0:
                mu = patch_tokens[idx].mean(dim=0)
            else:
                mu = patch_tokens[0]  # fallback

            protos.append(mu.unsqueeze(0)) # shape: [1, 1024]

        protos = torch.cat(protos, dim=0) # shape: [num_clusters, 1024]

        return protos

    def forward(self, x, is_training=True):

        # cls token (shape: [B, 1, 1024]), register token (shape: [B, 1, 1024]), patch token (shape: [B, 1, 1024, 1024])
        cls_tokens, register_tokens, patch_tokens = self.dinov3(x, use_lora=True) 

        # Global classification
        if self.use_deep_supervision == False or is_training == False:
            cls_tokens_24 = cls_tokens[:, -1]  # shape: [B, 1024]
            global_logits_24 = self.global_classifier(self.norm_global(cls_tokens_24)) # shape: [B, 1]
        else:
            cls_tokens_21 = cls_tokens[:, 0]  # shape: [B, 1024]
            cls_tokens_22 = cls_tokens[:, 1]  # shape: [B, 1024]
            cls_tokens_23 = cls_tokens[:, 2]  # shape: [B, 1024]
            cls_tokens_24 = cls_tokens[:,-1]  # shape: [B, 1024]
            global_logits_21 = self.global_classifier_21(self.norm_global_21(cls_tokens_21)) # shape: [B, 1]
            global_logits_22 = self.global_classifier_22(self.norm_global_22(cls_tokens_22)) # shape: [B, 1]
            global_logits_23 = self.global_classifier_23(self.norm_global_23(cls_tokens_23)) # shape: [B, 1]
            global_logits_24 = self.global_classifier(self.norm_global(cls_tokens_24)) # shape: [B, 1]


        # Patch-level classification
        if self.use_deep_supervision == False or is_training == False: 
            patch_tokens_24 = patch_tokens[:, -1]  # shape: [B, 1024, 1024]
        else:
            patch_tokens_21 = patch_tokens[:, 0]  # shape: [B, 1024, 1024]
            patch_tokens_22 = patch_tokens[:, 1]  # shape: [B, 1024, 1024]
            patch_tokens_23 = patch_tokens[:, 2]  # shape: [B, 1024, 1024]
            patch_tokens_24 = patch_tokens[:,-1]  # shape: [B, 1024, 1024]


        aggregated_patch_tokens_24, weak_patch_logits_24, rest_patch_logits_24, patch_logits = self.patch_classifier_reducer(patch_tokens_24, MIL_learning=True, reduction='mean', return_instance_logits=True)  # shape: [B, 1024], [B]
        if self.use_deep_supervision:
            aggregated_patch_tokens_21, weak_patch_logits_21, rest_patch_logits_21, patch_logits_21 = self.patch_classifier_reducer_21(patch_tokens_21, MIL_learning=True, reduction='mean', return_instance_logits=True)  # shape: [B, 1024], [B]
            aggregated_patch_tokens_22, weak_patch_logits_22, rest_patch_logits_22, patch_logits_22 = self.patch_classifier_reducer_22(patch_tokens_22, MIL_learning=True, reduction='mean', return_instance_logits=True)  # shape: [B, 1024], [B]
            aggregated_patch_tokens_23, weak_patch_logits_23, rest_patch_logits_23, patch_logits_23 = self.patch_classifier_reducer_23(patch_tokens_23, MIL_learning=True, reduction='mean', return_instance_logits=True)  # shape: [B, 1024], [B]    

        patch_logits_24 = self.patch_classifier(self.norm_patch(aggregated_patch_tokens_24)) # shape: [B, 1]
        if self.use_deep_supervision:
            patch_logits_21 = self.patch_classifier_21(self.norm_patch_21(aggregated_patch_tokens_21)) # shape: [B, 1]
            patch_logits_22 = self.patch_classifier_22(self.norm_patch_22(aggregated_patch_tokens_22)) # shape: [B, 1]
            patch_logits_23 = self.patch_classifier_23(self.norm_patch_23(aggregated_patch_tokens_23)) # shape: [B, 1]


        # Segment-level classification
        with torch.no_grad():
            _,_, patch_tokens_no_lora = self.dinov3(x, use_lora=False) 
            patch_tokens_24_no_lora = patch_tokens_no_lora[:, -1]  # shape: [B, 1024, 1024]

        B, N, D = patch_tokens_24.shape

        aggregated_segment_tokens_24, weak_segment_logits_24, rest_segment_logits_24 = [], [], []
        if self.use_deep_supervision:
            aggregated_segment_tokens_21, weak_segment_logits_21, rest_segment_logits_21 = [], [], []
            aggregated_segment_tokens_22, weak_segment_logits_22, rest_segment_logits_22 = [], [], []
            aggregated_segment_tokens_23, weak_segment_logits_23, rest_segment_logits_23 = [], [], []
        
        for b in range(B):
            with torch.no_grad():
                cluster_labels, _ = vit_patch_clustering(patch_tokens_24_no_lora[b], tau=0.9) # shape: [1024,]

            cluster_prototype_tokens = self.layerwise_segment(patch_tokens_24[b], cluster_labels) # shape: [num_clusters, 1024]
            if self.use_deep_supervision:
                cluster_prototype_tokens_21 = self.layerwise_segment(patch_tokens_21[b], cluster_labels) # shape: [num_clusters, 1024]
                cluster_prototype_tokens_22 = self.layerwise_segment(patch_tokens_22[b], cluster_labels) # shape: [num_clusters, 1024]
                cluster_prototype_tokens_23 = self.layerwise_segment(patch_tokens_23[b], cluster_labels) # shape: [num_clusters, 1024]

            aggregated_segment_token, weak_segment_logit, rest_segment_logit, weak_segment_instance_logit = self.segment_classifier_reducer(cluster_prototype_tokens.unsqueeze(0), MIL_learning=True, reduction='mean', return_instance_logits=True)  # shape: [1, 1024], [1], [1], [1, num_clusters]

            if self.use_deep_supervision:
                aggregated_segment_token_21, weak_segment_logit_21, rest_segment_logit_21, weak_segment_instance_logit_21 = self.segment_classifier_reducer_21(cluster_prototype_tokens_21.unsqueeze(0), MIL_learning=True, reduction='mean', return_instance_logits=True)  # shape: [1, 1024], [1], [1], [1, num_clusters]
                aggregated_segment_token_22, weak_segment_logit_22, rest_segment_logit_22, weak_segment_instance_logit_22 = self.segment_classifier_reducer_22(cluster_prototype_tokens_22.unsqueeze(0), MIL_learning=True, reduction='mean', return_instance_logits=True)  # shape: [1, 1024], [1], [1], [1, num_clusters]
                aggregated_segment_token_23, weak_segment_logit_23, rest_segment_logit_23, weak_segment_instance_logit_23 = self.segment_classifier_reducer_23(cluster_prototype_tokens_23.unsqueeze(0), MIL_learning=True, reduction='mean', return_instance_logits=True)  # shape: [1, 1024], [1], [1], [1, num_clusters]

            aggregated_segment_tokens_24.append(aggregated_segment_token)
            weak_segment_logits_24.append(weak_segment_logit)
            rest_segment_logits_24.append(rest_segment_logit)

            if self.use_deep_supervision:
                aggregated_segment_tokens_21.append(aggregated_segment_token_21)
                weak_segment_logits_21.append(weak_segment_logit_21)
                rest_segment_logits_21.append(rest_segment_logit_21)

                aggregated_segment_tokens_22.append(aggregated_segment_token_22)
                weak_segment_logits_22.append(weak_segment_logit_22)
                rest_segment_logits_22.append(rest_segment_logit_22)

                aggregated_segment_tokens_23.append(aggregated_segment_token_23)
                weak_segment_logits_23.append(weak_segment_logit_23)
                rest_segment_logits_23.append(rest_segment_logit_23)

        aggregated_segment_tokens_24 = torch.stack(aggregated_segment_tokens_24, dim=0).squeeze(dim=1)  # shape: [B, 1024]
        weak_segment_logits_24 = torch.stack(weak_segment_logits_24, dim=0)  # shape: [B]
        rest_segment_logits_24 = torch.stack(rest_segment_logits_24, dim=0)  # shape: [B]

        if self.use_deep_supervision:

            aggregated_segment_tokens_21 = torch.stack(aggregated_segment_tokens_21, dim=0).squeeze(dim=1)  # shape: [B, 1024]
            weak_segment_logits_21 = torch.stack(weak_segment_logits_21, dim=0)  # shape: [B]
            rest_segment_logits_21 = torch.stack(rest_segment_logits_21, dim=0)  # shape: [B]

            aggregated_segment_tokens_22 = torch.stack(aggregated_segment_tokens_22, dim=0).squeeze(dim=1)  # shape: [B, 1024]
            weak_segment_logits_22 = torch.stack(weak_segment_logits_22, dim=0)  # shape: [B]
            rest_segment_logits_22 = torch.stack(rest_segment_logits_22, dim=0)  # shape: [B]

            aggregated_segment_tokens_23 = torch.stack(aggregated_segment_tokens_23, dim=0).squeeze(dim=1)  # shape: [B, 1024]
            weak_segment_logits_23 = torch.stack(weak_segment_logits_23, dim=0)  # shape: [B]
            rest_segment_logits_23 = torch.stack(rest_segment_logits_23, dim=0)  # shape: [B]
            

        segment_logits_24 = self.segment_classifier(self.norm_segment(aggregated_segment_tokens_24)) # shape: [B, 1]

        if self.use_deep_supervision:
            segment_logits_21 = self.segment_classifier_21(self.norm_segment_21(aggregated_segment_tokens_21)) # shape: [B, 1]
            segment_logits_22 = self.segment_classifier_22(self.norm_segment_22(aggregated_segment_tokens_22)) # shape: [B, 1]
            segment_logits_23 = self.segment_classifier_23(self.norm_segment_23(aggregated_segment_tokens_23)) # shape: [B, 1]
        
        
        overall_feature_24 = torch.cat([
            self.norm_global(cls_tokens_24),
            self.norm_patch(aggregated_patch_tokens_24),
            self.norm_segment(aggregated_segment_tokens_24)
        ], dim=1) # shape: [B, 1024*3]

        main_logits_24 = self.main_classifier(overall_feature_24) # shape: [B, 1]

        if not self.use_deep_supervision:
            logits_dict = {
                "main_logits_24": main_logits_24,
                "global_logits_24": global_logits_24,
                "patch_logits_24": patch_logits_24,
                "segment_logits_24": segment_logits_24,
                "weak_patch_logits_24": weak_patch_logits_24,
                "weak_segment_logits_24": weak_segment_logits_24,
                "rest_patch_logits_24":rest_patch_logits_24,
                "rest_segment_logits_24": rest_segment_logits_24
            }
        else:
            logits_dict = {
                "main_logits_24": main_logits_24,
                "global_logits_24": global_logits_24,
                "patch_logits_24": patch_logits_24,
                "segment_logits_24": segment_logits_24,
                "weak_patch_logits_24": weak_patch_logits_24,
                "weak_segment_logits_24": weak_segment_logits_24,
                "rest_patch_logits_24":rest_patch_logits_24,
                "rest_segment_logits_24": rest_segment_logits_24,

                "global_logits_23": global_logits_23,
                "patch_logits_23": patch_logits_23,
                "segment_logits_23": segment_logits_23,
                "weak_patch_logits_23": weak_patch_logits_23,
                "weak_segment_logits_23": weak_segment_logits_23,
                "rest_patch_logits_23":rest_patch_logits_23,
                "rest_segment_logits_23": rest_segment_logits_23,

                "global_logits_22": global_logits_22,
                "patch_logits_22": patch_logits_22,
                "segment_logits_22": segment_logits_22,
                "weak_patch_logits_22": weak_patch_logits_22,
                "weak_segment_logits_22": weak_segment_logits_22,
                "rest_patch_logits_22":rest_patch_logits_22,
                "rest_segment_logits_22": rest_segment_logits_22,

                "global_logits_21": global_logits_21,
                "patch_logits_21": patch_logits_21,
                "segment_logits_21": segment_logits_21,
                "weak_patch_logits_21": weak_patch_logits_21,
                "weak_segment_logits_21": weak_segment_logits_21,
                "rest_patch_logits_21":rest_patch_logits_21,
                "rest_segment_logits_21": rest_segment_logits_21
            }

        # 除了原本为[B]的logits，将其他所有形状为[B,1]的logits展平为[B]
        for logits_name, logits in logits_dict.items():
            if logits.dim() == 2:
                logits_dict[logits_name] = logits.squeeze(1)  # shape: [B]

        if not self.use_deep_supervision:
            tokens_dict = {
                "cls_tokens_24": cls_tokens_24,
                "aggregated_patch_tokens_24": aggregated_patch_tokens_24,
                "aggregated_segment_tokens_24": aggregated_segment_tokens_24
            }
        else:
            tokens_dict = {
                "cls_tokens_24": cls_tokens_24,
                "aggregated_patch_tokens_24": aggregated_patch_tokens_24,
                "aggregated_segment_tokens_24": aggregated_segment_tokens_24,

                "cls_tokens_23": cls_tokens_23,
                "aggregated_patch_tokens_23": aggregated_patch_tokens_23,
                "aggregated_segment_tokens_23": aggregated_segment_tokens_23,

                "cls_tokens_22": cls_tokens_22,
                "aggregated_patch_tokens_22": aggregated_patch_tokens_22,
                "aggregated_segment_tokens_22": aggregated_segment_tokens_22,

                "cls_tokens_21": cls_tokens_21,
                "aggregated_patch_tokens_21": aggregated_patch_tokens_21,
                "aggregated_segment_tokens_21": aggregated_segment_tokens_21
            }

        if is_training == False:
            return main_logits_24.squeeze(1), global_logits_24.squeeze(1), patch_logits_24.squeeze(1), segment_logits_24.squeeze(1)
        else:
            return logits_dict, tokens_dict