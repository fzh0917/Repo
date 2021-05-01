from typing import Dict

from ..transformer_base import TRACK_TRANSFORMERS, TransformerBase
from videoanalyst.data.utils.target_image_crop import crop


@TRACK_TRANSFORMERS.register
class RandomCropTransformer(TransformerBase):

    default_hyper_params = dict(
        max_scale=0.3,
        max_shift=0.4,
        q_size=289,
        num_memory_frames=0,
        search_area_factor=0.0,
        phase_mode="train",
    )

    def __init__(self, seed: int = 0) -> None:
        super(RandomCropTransformer, self).__init__(seed=seed)

    def __call__(self, sampled_data: Dict) -> Dict:
        data1 = sampled_data["data1"]
        data2 = sampled_data["data2"]
        sampled_data["data1"] = {}
        nmf = self._hyper_params['num_memory_frames']
        search_area_factor = self._hyper_params['search_area_factor']
        q_size = self._hyper_params['q_size']
        for i in range(nmf):
            im_memory, bbox_memory = data1["image_{}".format(i)], data1["anno_{}".format(i)]
            im_m, bbox_m, _ = crop(im_memory, bbox_memory, search_area_factor, q_size,
                                   config=self._hyper_params, rng=self._state["rng"])
            sampled_data["data1"]['image_{}'.format(i)] = im_m
            sampled_data["data1"]['anno_{}'.format(i)] = bbox_m
        im_query, bbox_query = data2["image"], data2["anno"]
        im_q, bbox_q, _ = crop(im_query, bbox_query, search_area_factor, q_size,
                               config=self._hyper_params, rng=self._state["rng"])
        sampled_data["data2"] = dict(image=im_q, anno=bbox_q)

        return sampled_data
