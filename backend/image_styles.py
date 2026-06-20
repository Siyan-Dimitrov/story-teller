"""Backend-owned image style registry."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_STYLE_ID = "animated_storybook"
DEFAULT_STYLE_PROMPT = (
    "hand-drawn 2D animated storybook still, non-photorealistic, clean line art, "
    "soft cel shading, painted background, expressive silhouettes, theatrical "
    "composition, rich color palette, cinematic lighting, symbolic folklore mood"
)


@dataclass(frozen=True)
class ImageStyle:
    id: str
    label: str
    description: str
    prompt: str
    default_lora_keys: tuple[str, ...] = ()

    @property
    def supports_loras(self) -> bool:
        return bool(self.default_lora_keys)

    def public_dict(self) -> dict:
        data = {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "supports_loras": self.supports_loras,
        }
        if self.default_lora_keys:
            data["default_lora_keys"] = list(self.default_lora_keys)
        return data


IMAGE_STYLES: tuple[ImageStyle, ...] = (
    ImageStyle(
        id="animated_storybook",
        label="Animated Storybook",
        description="Hand-drawn 2D animated fairy-tale look.",
        prompt=DEFAULT_STYLE_PROMPT,
        default_lora_keys=("storybook",),
    ),
    ImageStyle(
        id="gothic_folklore",
        label="Gothic Folklore",
        description="Moody gothic folklore with theatrical shadows.",
        prompt=(
            "whimsical gothic folklore illustration, crooked architecture, moonlit "
            "silhouettes, ornate storybook detail, expressive character shapes, "
            "symbolic drama, non-graphic, non-photorealistic, cinematic shadow play"
        ),
        default_lora_keys=("tim_burton", "dark_gothic"),
    ),
    ImageStyle(
        id="dark_fantasy_painting",
        label="Dark Fantasy Painting",
        description="Painterly dark fantasy with rich atmosphere.",
        prompt=(
            "dark fantasy concept painting, dramatic composition, deep shadows, "
            "magical atmosphere, painterly brush texture, stylized characters, "
            "non-photorealistic, rich environmental storytelling"
        ),
        default_lora_keys=("dark_gothic", "concept_art"),
    ),
    ImageStyle(
        id="ink_watercolor",
        label="Ink Watercolor",
        description="Pen-and-ink storybook with watercolor washes.",
        prompt=(
            "fine ink linework, watercolor texture, parchment warmth, delicate "
            "storybook composition, expressive but restrained mood, hand-painted "
            "folklore illustration, non-photorealistic"
        ),
        default_lora_keys=("storybook",),
    ),
    ImageStyle(
        id="paper_cutout_theatre",
        label="Paper Cutout Theatre",
        description="Layered paper theatre and shadow-box style.",
        prompt=(
            "layered paper cutout theatre, handmade paper textures, flat stylized "
            "shapes, depth through overlapping silhouettes, warm stage lighting, "
            "shadow-box folklore scene, non-photorealistic"
        ),
    ),
    ImageStyle(
        id="vintage_animation",
        label="Vintage Animation",
        description="Classic painted animation cel look.",
        prompt=(
            "vintage 2D animation cel, painted background, simplified expressive "
            "characters, clean outlines, soft film grain, cinematic staging, "
            "non-photorealistic animated folklore frame"
        ),
        default_lora_keys=("painterly_illustration",),
    ),
    ImageStyle(
        id="storybook_sketch",
        label="Storybook Sketch",
        description="Gentle sketchy children's-book illustration.",
        prompt=(
            "soft graphite and pastel storybook sketch, gentle linework, simplified "
            "features, warm palette, safe symbolic emotion, hand-drawn folklore "
            "illustration, non-photorealistic"
        ),
        default_lora_keys=("children_sketch",),
    ),
    ImageStyle(
        id="surreal_folklore",
        label="Surreal Folklore",
        description="Dreamlike symbolic fairy-tale surrealism.",
        prompt=(
            "surreal folklore illustration, dreamlike scale, symbolic props, pale "
            "porcelain-like colors, strange but elegant atmosphere, stylized "
            "characters, non-realistic, non-graphic"
        ),
        default_lora_keys=("mark_ryden",),
    ),
    ImageStyle(
        id="golden_fable",
        label="Golden Fable",
        description="Warm glowing fable with golden light.",
        prompt=(
            "golden-hour fable illustration, luminous dust, warm painted background, "
            "hopeful dramatic lighting, rich color harmony, stylized storybook "
            "characters, non-photorealistic"
        ),
        default_lora_keys=("golden_atmosphere",),
    ),
    ImageStyle(
        id="cinematic_concept",
        label="Cinematic Concept",
        description="Polished animated concept-art frame.",
        prompt=(
            "stylized cinematic concept art, strong composition, painterly detail, "
            "animated-film mood, non-photorealistic characters, atmospheric lighting, "
            "rich environment design, symbolic folklore drama"
        ),
        default_lora_keys=("concept_art",),
    ),
)

STYLE_BY_ID = {style.id: style for style in IMAGE_STYLES}


def public_styles_response() -> dict:
    return {
        "styles": [style.public_dict() for style in IMAGE_STYLES],
        "default_style_id": DEFAULT_STYLE_ID,
    }


def get_style(style_id: str | None) -> ImageStyle | None:
    if not style_id:
        return None
    return STYLE_BY_ID.get(style_id)


def resolve_style_prompt(
    *,
    style_id: str | None = None,
    custom_style_prompt: str | None = None,
    legacy_style_prompt: str | None = None,
) -> str:
    custom = (custom_style_prompt or "").strip()
    if custom:
        return custom

    style = get_style(style_id)
    if style:
        return style.prompt

    legacy = (legacy_style_prompt or "").strip()
    if legacy:
        return legacy

    default_style = STYLE_BY_ID[DEFAULT_STYLE_ID]
    return default_style.prompt


def resolve_style_loras(
    *,
    style_id: str | None = None,
    request_lora_keys: list[str] | None = None,
) -> list[str] | None:
    if request_lora_keys is not None:
        return request_lora_keys

    style = get_style(style_id)
    if style and style.default_lora_keys:
        return list(style.default_lora_keys)

    return None
