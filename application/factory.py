from template_matching.SAD import SADclass
from template_matching.SSD import SSDclass
from template_matching.NCC import NCCclass


def create(type_class, use_mask):
    if type_class == "SAD":
        return SADclass(use_mask)
    if type_class == "SSD":
        return SSDclass(use_mask)
    if type_class == "NCC":
        return NCCclass(use_mask)
