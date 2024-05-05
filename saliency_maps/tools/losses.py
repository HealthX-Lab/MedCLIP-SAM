import torch.nn as nn
import torch

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def loss(self, img_features, text_features, labels):
        """
        Args: 
            img_features: [b, num_proposals, img_feature_size]
            text_features: [b, num_phrases, text_feature_size]
            labels: [b_caption, b_img] caption level label
        Returns:
            Scalar, loss.
        """
        # [b_text, num_phrase, feature_size], [b_img, num_proposals, feaeture_size] --> [b_text, b_img, num_phrase, num_proposal]
        att_matrix = torch.einsum('btf, dif -> bdti', text_features, img_features) 
        argmax_box = att_matrix.max(dim=3).values
        score_matrix =  argmax_box.mean(dim=2) #[b_caption, b_img]

        score_exp = torch.exp(score_matrix/self.temperature)
        