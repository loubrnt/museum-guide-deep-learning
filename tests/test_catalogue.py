import pandas as pd
import pytest

from museum_guide.catalogue import Catalogue, fold, key, repair_slug, stem, tokens

GENRES = ["cityscape", "landscape", "portrait", "religious_painting", "still_life"]
ARTISTS = [
    "claude-monet", "pierre-auguste-renoir", "ilya-mashkov", "edouard-cortes", "juan-gris", "henri-fantin-latour",
    "vincent-van-gogh", "hans-holbein-the-younger", "joaquã\xadn-sorolla", "konstantin-korovin", "salvador-dali",
    "fernand-leger", "mstislav-dobuzhinsky", "canaletto",
]
STYLES = [
    "Impressionism", "Post_Impressionism", "Cubism", "Synthetic_Cubism", "Analytical_Cubism", "Baroque", "Northern_Renaissance",
    "Naive_Art_Primitivism", "Realism", "Contemporary_Realism", "Romanticism", "Rococo", "Art_Nouveau", "Pop_Art", "Ukiyo_e",
    "Early_Renaissance", "High_Renaissance",
]


@pytest.fixture(scope="module")
def meta():
    rows = [{"artist": a, "style": STYLES[i % len(STYLES)], "label": GENRES[i % 5]} for i, a in enumerate(ARTISTS)]
    rows += [{"artist": "Unknown Artist", "style": s, "label": g} for s in STYLES for g in GENRES]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def cat(meta):
    return Catalogue.from_metadata(meta)


# ---------------------------------------------------------------------------------------- text helpers
def test_fold_removes_accents_and_keeps_positions():
    text = "Édouard Cortès, Zinaïda"
    assert fold(text) == "edouard cortes, zinaida"
    assert len(fold(text)) == len(text)


def test_fold_handles_decomposed_input():
    assert fold("Édouard") == "edouard"


def test_stem_drops_a_plural_s_only_from_long_words():
    assert stem("portraits") == "portrait"
    assert stem("impressionnistes") == "impressionniste"
    assert stem("bus") == "bus"  # too short
    assert stem("glass") == "glass"  # double s


def test_tokens_split_hyphens_and_apostrophes_and_give_character_spans():
    text = "d'Alfred Fantin-Latour"
    out = tokens(text)
    assert [t for t, _, _ in out] == ["d", "alfred", "fantin", "latour"]
    assert [text[s:e] for _, s, e in out] == ["d", "Alfred", "Fantin", "Latour"]


def test_key_is_the_normalised_word_tuple():
    assert key("Post-Impressionist") == ("post", "impressionist")
    assert key("Nature Mortes") == ("nature", "morte")


def test_repair_slug_fixes_utf8_read_as_latin1_after_lowercasing():
    assert repair_slug("joaquã\xadn-sorolla") == "joaquín-sorolla"
    assert repair_slug("claude-monet") == "claude-monet"
    assert repair_slug("edouard-cortes") == "edouard-cortes"


# ---------------------------------------------------------------------------------------- artists
@pytest.mark.parametrize(
    "query, artist",
    [
        ("paintings by Claude Monet", "claude-monet"),
        ("a Monet landscape", "claude-monet"),
        ("MONET", "claude-monet"),
        ("des tableaux de Claude Monet", "claude-monet"),
        ("a portrait by Renoir", "pierre-auguste-renoir"),
        ("Pierre-Auguste Renoir", "pierre-auguste-renoir"),
        ("des tableaux d'Édouard Cortès", "edouard-cortes"),
        ("des tableaux d'Edouard Cortes", "edouard-cortes"),
        ("des tableaux d'Ilia Machkov", "ilya-mashkov"),  # French transliteration of a Russian name
        ("works by Ilya Mashkov", "ilya-mashkov"),
        ("Constantin Korovine", "konstantin-korovin"),
        ("Mstislav Doboujinski", "mstislav-dobuzhinsky"),
        ("Fantin-Latour flowers", "henri-fantin-latour"),
        ("Henri Fantin-Latour", "henri-fantin-latour"),
        ("a work by Vincent van Gogh", "vincent-van-gogh"),
        ("van Gogh", "vincent-van-gogh"),
        ("Hans Holbein", "hans-holbein-the-younger"),
        ("Hans Holbein the Younger", "hans-holbein-the-younger"),
        ("Joaquín Sorolla", "joaquã\xadn-sorolla"),  # the slug keeps its raw form: it is what the metadata contains
        ("paintings by Sorolla", "joaquã\xadn-sorolla"),
        ("Canaletto", "canaletto"),
    ],
)
def test_artists_are_found_in_english_and_french_with_accents_and_transliterations(cat, query, artist):
    parsed = cat.parse(query)
    assert parsed.artists == (artist,), parsed.describe()


def test_the_readable_label_of_a_repaired_slug_is_clean(cat):
    assert cat.parse("Sorolla").detections[0].label == "Joaquín Sorolla"


def test_a_surname_that_is_a_common_word_needs_a_cue(cat):
    assert cat.parse("a gris sky").artists == ()
    assert cat.parse("un ciel gris").artists == ()
    assert cat.parse("paintings by Gris").artists == ("juan-gris",)
    assert cat.parse("un tableau de Gris").artists == ("juan-gris",)
    assert cat.parse("a work by Leger").artists == ("fernand-leger",)
    assert cat.parse("un léger sourire").artists == ()
    assert cat.parse("Juan Gris").artists == ("juan-gris",)  # the full name needs no cue


def test_a_short_surname_needs_a_cue(cat):
    assert cat.parse("Dali").artists == ()
    assert cat.parse("by Dali").artists == ("salvador-dali",)
    assert cat.parse("Salvador Dali").artists == ("salvador-dali",)


def test_first_names_alone_never_match(cat):
    assert cat.parse("a painting by Claude").artists == ()
    assert cat.parse("Henri").artists == ()


def test_two_artists_make_a_union(cat):
    assert set(cat.parse("paintings by Monet and Renoir").artists) == {"claude-monet", "pierre-auguste-renoir"}


def test_unknown_artist_is_not_a_name(cat):
    assert cat.parse("Unknown Artist").artists == ()


def test_a_name_shared_by_two_artists_is_dropped_but_full_names_still_work():
    meta = pd.DataFrame(
        {"artist": ["jean-martin", "pierre-martin"], "style": ["Baroque"] * 2, "label": ["portrait"] * 2}
    )
    cat = Catalogue.from_metadata(meta)
    assert ("martin",) in cat.ambiguous
    assert cat.parse("by Martin").artists == ()  # no match is better than a wrong filter
    assert cat.parse("Jean Martin").artists == ("jean-martin",)
    assert cat.parse("Pierre Martin").artists == ("pierre-martin",)


# ---------------------------------------------------------------------------------------- styles and genres
def test_a_cubist_question_covers_the_three_cubism_labels(cat):
    parsed = cat.parse("a Cubist still life")
    assert set(parsed.styles) == {"Cubism", "Synthetic_Cubism", "Analytical_Cubism"}
    assert parsed.genres == ("still_life",)


def test_a_sub_style_named_on_its_own_keeps_to_itself(cat):
    assert cat.parse("Synthetic Cubism").styles == ("Synthetic_Cubism",)
    assert set(cat.parse("Cubism").styles) == {"Cubism", "Synthetic_Cubism", "Analytical_Cubism"}


def test_post_impressionism_is_not_read_as_impressionism(cat):
    assert cat.parse("a Post-Impressionist landscape").styles == ("Post_Impressionism",)
    assert cat.parse("post impressionism").styles == ("Post_Impressionism",)
    assert cat.parse("an Impressionist landscape").styles == ("Impressionism",)


@pytest.mark.parametrize(
    "query, style",
    [
        ("un paysage impressionniste", "Impressionism"),
        ("des peintures baroques", "Baroque"),
        ("un portrait rococo", "Rococo"),
        ("a Romantic landscape", "Romanticism"),
        ("le romantisme", "Romanticism"),
        ("un portrait réaliste", "Realism"),
        ("a contemporary realist painting", "Contemporary_Realism"),
        ("une peinture naïve", "Naive_Art_Primitivism"),
        ("naive art", "Naive_Art_Primitivism"),
        ("Art nouveau", "Art_Nouveau"),
        ("un portrait pop art", "Pop_Art"),
        ("ukiyo-e prints", "Ukiyo_e"),
        ("un portrait de la Renaissance nordique", "Northern_Renaissance"),
        ("la Renaissance du Nord", "Northern_Renaissance"),
        ("a Northern Renaissance portrait", "Northern_Renaissance"),
        ("une peinture de la Haute Renaissance", "High_Renaissance"),
        ("un portrait de la première Renaissance", "Early_Renaissance"),
    ],
)
def test_styles_in_english_and_french(cat, query, style):
    assert style in cat.parse(query).styles, cat.parse(query).describe()


def test_the_word_renaissance_alone_is_not_a_style(cat):
    assert cat.parse("a Renaissance painting").styles == ()


@pytest.mark.parametrize(
    "query, genre",
    [
        ("a city view", "cityscape"),
        ("a cityscape", "cityscape"),
        ("un paysage urbain", "cityscape"),
        ("des vues de ville", "cityscape"),
        ("an urban landscape", "cityscape"),
        ("a landscape", "landscape"),
        ("des paysages", "landscape"),
        ("portraits", "portrait"),
        ("a still life", "still_life"),
        ("a still-life", "still_life"),
        ("des natures mortes", "still_life"),
        ("a religious painting", "religious_painting"),
        ("des peintures religieuses", "religious_painting"),
        ("un tableau religieux", "religious_painting"),
    ],
)
def test_genres_in_english_and_french(cat, query, genre):
    assert cat.parse(query).genres == (genre,), cat.parse(query).describe()


def test_an_urban_landscape_is_not_also_a_landscape(cat):
    assert cat.parse("un paysage urbain impressionniste").genres == ("cityscape",)  # longest match wins


def test_a_style_missing_from_the_collection_is_not_in_the_vocabulary():
    meta = pd.DataFrame({"artist": ["claude-monet"], "style": ["Impressionism"], "label": ["landscape"]})
    cat = Catalogue.from_metadata(meta)
    assert cat.parse("a Baroque portrait").has_filters is False
    assert cat.parse("an Impressionist landscape").has_filters is True


# ---------------------------------------------------------------------------------------- no match is better than a wrong one
@pytest.mark.parametrize("query", ["a stormy sea", "a snowy winter village", "", "   ", "la mer agitée", "a vase of flowers", "the crucifixion of Christ"])
def test_a_question_without_catalogue_fields_gets_no_filter(cat, query):
    parsed = cat.parse(query)
    assert not parsed.has_filters
    assert parsed.detections == ()
    assert parsed.visual_text == query


def test_whole_words_only(cat):
    assert cat.parse("autoportrait").genres == ()
    assert cat.parse("a landscaper").genres == ()


# ---------------------------------------------------------------------------------------- the words left for CLIP
def test_remainder_keeps_only_what_the_catalogue_cannot_say(cat):
    parsed = cat.parse("a portrait by Renoir of a woman with a hat")
    assert parsed.genres == ("portrait",)
    assert parsed.artists == ("pierre-auguste-renoir",)
    assert parsed.remainder == "woman with a hat"
    assert parsed.visual_text == "woman with a hat"


def test_remainder_of_a_question_made_of_fields_only_is_empty_and_clip_gets_the_whole_question(cat):
    for query in ("paintings by Claude Monet", "a Cubist still life", "un paysage impressionniste", "des tableaux d'Édouard Cortès"):
        parsed = cat.parse(query)
        assert parsed.has_filters
        assert parsed.remainder == ""
        assert parsed.visual_text == query


def test_remainder_drops_the_cue_before_a_painter_and_trailing_stop_words(cat):
    assert cat.parse("a snowy street by Claude Monet").remainder == "snowy street"
    assert cat.parse("a Baroque religious painting with angels").remainder == "angels"
    assert cat.parse("a Romantic landscape with a stormy sea").remainder == "stormy sea"
    assert cat.parse("un paysage impressionniste avec une rivière").remainder == "rivière"


def test_detections_give_the_matched_text_and_its_span(cat):
    q = "a Cubist still life with a guitar"
    parsed = cat.parse(q)
    assert [(d.field, d.matched, q[d.start : d.end]) for d in parsed.detections] == [
        ("style", "Cubist", "Cubist"),
        ("genre", "still life", "still life"),
    ]


def test_describe_names_what_was_found(cat):
    assert cat.parse("a Cubist still life").describe() == "style: Cubist, genre: still life"
    assert cat.parse("a stormy sea").describe() == "no catalogue field found"


# ---------------------------------------------------------------------------------------- the mask
def test_mask_combines_the_fields_with_and(cat, meta):
    parsed = cat.parse("a Cubist still life")
    keep = cat.mask(meta, parsed)
    expected = meta["style"].isin(["Cubism", "Synthetic_Cubism", "Analytical_Cubism"]) & (meta["label"] == "still_life")
    assert keep.tolist() == expected.tolist()
    assert keep.any()


def test_mask_without_filters_keeps_everything(cat, meta):
    assert cat.mask(meta, cat.parse("a stormy sea")).all()


def test_mask_by_artist(cat, meta):
    keep = cat.mask(meta, cat.parse("Monet"))
    assert meta.loc[keep, "artist"].tolist() == ["claude-monet"]
