"""Short notices on the tradition a painting belongs to: what the guide says after recognising a genre or a style.

The brief asks the guide to recognise a painting's style or genre "and tell the story of that tradition". These are
the stories: two to four sentences per genre and per style, in French (the museum's language) and in English.

    from museum_guide.notices import notice_for, title_for
    notice_for("portrait", "fr")               # the text, which starts with the name of the tradition
    title_for("Post_Impressionism", "en")      # "Post-Impressionism"

THE TEXTS ARE DRAFTS written by an AI assistant. They stick to widely known general facts, but they must be read
by the team, and ideally by someone trained in art history, before they are shown to visitors.

Labels are the ones used in the project: the five genres of the WikiArt subset (`museum_guide.predict.CLASSES`) and
the styles that appear at least 15 times in its metadata (WikiArt spellings such as `Post_Impressionism`).
"""
from __future__ import annotations

import re

LANGS = ("fr", "en")

# canonical label -> {language: (title, text)}
_GENRES: dict[str, dict[str, tuple[str, str]]] = {
    "cityscape": {
        "fr": (
            "La vue de ville",
            "La vue de ville montre des rues, des places et des monuments. Au XVIIIe siècle, Canaletto a rendu célèbres "
            "les vues de Venise ; plus tard, les impressionnistes ont peint les boulevards modernes de Paris. Regardez "
            "la perspective, la lumière sur les façades et les petits personnages qui donnent vie au lieu.",
        ),
        "en": (
            "The cityscape",
            "The cityscape shows streets, squares and monuments. In the 18th century Canaletto made views of Venice "
            "famous; later the Impressionists painted the modern boulevards of Paris. Look at the perspective, the "
            "light on the façades and the small figures that bring the place to life.",
        ),
    },
    "landscape": {
        "fr": (
            "Le paysage",
            "Le paysage a longtemps servi de décor à des scènes religieuses ou historiques, avant de devenir un genre à "
            "part entière, notamment aux Pays-Bas au XVIIe siècle. Les romantiques puis les impressionnistes en ont fait "
            "un lieu d'émotion et de lumière. Regardez l'horizon, le ciel et la façon dont la lumière organise l'espace.",
        ),
        "en": (
            "The landscape",
            "For a long time the landscape was only a setting for religious or historical scenes, before it became a "
            "genre in its own right, notably in the Netherlands in the 17th century. Romantic and then Impressionist "
            "painters made it a place of emotion and light. Look at the horizon, the sky and how light organises space.",
        ),
    },
    "portrait": {
        "fr": (
            "Le portrait",
            "Le portrait représente une personne précise, pour garder son souvenir ou affirmer son rang. À la "
            "Renaissance, en Italie comme en Flandre, il s'attache à rendre des traits individuels ; plus tard, les "
            "peintres cherchent aussi à suggérer le caractère. Regardez le regard, les vêtements et les objets qui "
            "disent qui était le modèle.",
        ),
        "en": (
            "The portrait",
            "A portrait shows a particular person, to preserve their memory or to assert their rank. In the "
            "Renaissance, in Italy as in Flanders, it aimed at individual features; later, painters also tried to "
            "suggest character. Look at the gaze, the clothes and the objects that say who the sitter was.",
        ),
    },
    "religious_painting": {
        "fr": (
            "La peinture religieuse",
            "La peinture religieuse raconte des scènes de la Bible et des vies de saints. Elle a longtemps orné les "
            "églises et aidé les fidèles à comprendre et à prier. Cherchez les auréoles et les attributs qui permettent "
            "d'identifier les personnages, comme les clés de saint Pierre ou l'agneau.",
        ),
        "en": (
            "Religious painting",
            "Religious painting tells scenes from the Bible and the lives of saints. For centuries it decorated "
            "churches and helped the faithful to understand and to pray. Look for the halos and the attributes that "
            "identify the figures, such as the keys of Saint Peter or the lamb.",
        ),
    },
    "still_life": {
        "fr": (
            "La nature morte",
            "La nature morte met en scène des objets : fruits, fleurs, vaisselle, livres. Elle s'épanouit en Flandre et "
            "aux Pays-Bas au XVIIe siècle, où certaines compositions, les vanités, rappellent par un crâne ou un sablier "
            "que la vie est brève. Plus tard, Cézanne y cherche la forme et la couleur.",
        ),
        "en": (
            "The still life",
            "A still life stages objects: fruit, flowers, tableware, books. It flourished in Flanders and the "
            "Netherlands in the 17th century, where some compositions, the vanitas pictures, remind us with a skull or "
            "an hourglass that life is short. Later, Cézanne used it to explore form and colour.",
        ),
    },
}

_STYLES: dict[str, dict[str, tuple[str, str]]] = {
    "Impressionism": {
        "fr": (
            "L'impressionnisme",
            "L'impressionnisme naît à Paris dans les années 1860-1870. Son nom vient d'« Impression, soleil levant » de "
            "Monet, montré à l'exposition de 1874. Ces peintres travaillent souvent en plein air et cherchent à saisir "
            "la lumière et l'instant par des touches rapides et visibles. Pensez à Monet, Renoir, Pissarro, Morisot.",
        ),
        "en": (
            "Impressionism",
            "Impressionism arose in Paris in the 1860s and 1870s. Its name comes from Monet's \"Impression, Sunrise\", "
            "shown at the 1874 exhibition. These painters often worked outdoors and tried to capture light and the "
            "passing moment with quick, visible brushstrokes. Think of Monet, Renoir, Pissarro and Morisot.",
        ),
    },
    "Post_Impressionism": {
        "fr": (
            "Le post-impressionnisme",
            "Le post-impressionnisme désigne des peintres actifs à partir des années 1880 qui partent de "
            "l'impressionnisme pour aller plus loin. Cézanne cherche la structure, Seurat la méthode, Van Gogh et "
            "Gauguin l'expression et le symbole. Regardez les couleurs intenses et les formes simplifiées, qui comptent "
            "plus que la ressemblance.",
        ),
        "en": (
            "Post-Impressionism",
            "Post-Impressionism covers painters active from the 1880s who start from Impressionism and go further. "
            "Cézanne looks for structure, Seurat for method, Van Gogh and Gauguin for expression and symbol. Look at "
            "the intense colours and the simplified forms, which matter more than likeness.",
        ),
    },
    "Realism": {
        "fr": (
            "Le réalisme",
            "Le réalisme apparaît en France vers le milieu du XIXe siècle. Courbet affirme qu'il faut peindre la vie "
            "réelle sans l'idéaliser : paysans, ouvriers, scènes ordinaires, parfois sur de très grandes toiles. Millet "
            "peint le monde rural. Regardez des sujets modestes, traités avec sérieux.",
        ),
        "en": (
            "Realism",
            "Realism appeared in France around the middle of the 19th century. Courbet insisted on painting real life "
            "without idealising it: peasants, workers, ordinary scenes, sometimes on very large canvases. Millet "
            "painted the countryside. Look for humble subjects treated with seriousness.",
        ),
    },
    "Expressionism": {
        "fr": (
            "L'expressionnisme",
            "L'expressionnisme se développe au début du XXe siècle, surtout en Allemagne, avec des groupes comme Die "
            "Brücke à Dresde. Ces artistes déforment les formes et emploient des couleurs violentes pour exprimer une "
            "émotion intérieure plutôt que l'apparence des choses. Regardez les contours appuyés et les couleurs qui ne "
            "copient pas la nature.",
        ),
        "en": (
            "Expressionism",
            "Expressionism developed in the early 20th century, above all in Germany, with groups such as Die Brücke "
            "in Dresden. These artists distort forms and use violent colours to express inner emotion rather than the "
            "appearance of things. Look at the heavy outlines and the colours that do not copy nature.",
        ),
    },
    "Romanticism": {
        "fr": (
            "Le romantisme",
            "Le romantisme se répand en Europe de la fin du XVIIIe siècle jusqu'au milieu du XIXe. Il met en avant "
            "l'émotion, l'imagination et la puissance de la nature, face à l'ordre et à la raison. Delacroix, Turner et "
            "Friedrich en sont des figures majeures. Regardez les ciels dramatiques et les sentiments exacerbés.",
        ),
        "en": (
            "Romanticism",
            "Romanticism spread across Europe from the late 18th century to the middle of the 19th. It stresses "
            "emotion, imagination and the power of nature, against order and reason. Delacroix, Turner and Friedrich "
            "are major figures. Look at the dramatic skies and the heightened feelings.",
        ),
    },
    "Baroque": {
        "fr": (
            "Le baroque",
            "Le baroque naît en Italie vers 1600 et domine le XVIIe siècle européen. Il cherche à frapper et à émouvoir "
            "par le mouvement, le drame et de forts contrastes d'ombre et de lumière, que l'on appelle clair-obscur. "
            "Caravage, Rubens, Rembrandt ou Velázquez en sont les grands noms.",
        ),
        "en": (
            "Baroque",
            "Baroque arose in Italy around 1600 and dominated 17th-century Europe. It sets out to strike and move the "
            "viewer through movement, drama and strong contrasts of light and shadow, known as chiaroscuro. Caravaggio, "
            "Rubens, Rembrandt and Velázquez are among its great names.",
        ),
    },
    "Northern_Renaissance": {
        "fr": (
            "La Renaissance du Nord",
            "La Renaissance du Nord désigne la peinture des XVe et XVIe siècles au nord des Alpes : Flandre, "
            "Pays-Bas, Allemagne. Elle se distingue par un emploi virtuose de la peinture à l'huile, un détail "
            "minutieux et des objets du quotidien chargés de sens. Van Eyck, Dürer, Bruegel et Holbein en sont des "
            "figures marquantes.",
        ),
        "en": (
            "The Northern Renaissance",
            "The Northern Renaissance names the painting of the 15th and 16th centuries north of the Alps: Flanders, "
            "the Netherlands, Germany. It stands out for its masterly use of oil paint, its minute detail and everyday "
            "objects loaded with meaning. Van Eyck, Dürer, Bruegel and Holbein are leading figures.",
        ),
    },
    "Cubism": {
        "fr": (
            "Le cubisme",
            "Le cubisme est inventé à Paris vers 1907-1914 par Picasso et Braque. Les objets sont décomposés en "
            "facettes géométriques et montrés sous plusieurs points de vue à la fois, ce qui rompt avec la perspective "
            "traditionnelle. Regardez les formes anguleuses et les objets qu'il faut reconstituer.",
        ),
        "en": (
            "Cubism",
            "Cubism was invented in Paris around 1907-1914 by Picasso and Braque. Objects are broken into geometric "
            "facets and shown from several viewpoints at once, breaking with traditional perspective. Look at the "
            "angular shapes and the objects you have to piece back together.",
        ),
    },
    "Synthetic_Cubism": {
        "fr": (
            "Le cubisme synthétique",
            "Le cubisme synthétique est la seconde phase du cubisme, à partir de 1912 environ. Les formes deviennent "
            "plus simples, plus plates et plus colorées, et les artistes introduisent le collage de papiers. Juan "
            "Gris, Braque et Picasso le pratiquent. Regardez les aplats et les imitations de bois ou de journal.",
        ),
        "en": (
            "Synthetic Cubism",
            "Synthetic Cubism is the second phase of Cubism, from about 1912. Forms become simpler, flatter and more "
            "colourful, and artists introduce paper collage. Juan Gris, Braque and Picasso practised it. Look at the "
            "flat areas of colour and the imitations of wood or newsprint.",
        ),
    },
    "Early_Renaissance": {
        "fr": (
            "La première Renaissance",
            "La première Renaissance correspond au XVe siècle en Italie, surtout à Florence. Les artistes redécouvrent "
            "l'Antiquité, étudient le corps humain et mettent au point la perspective linéaire pour créer un espace "
            "crédible. Masaccio, Fra Angelico, Botticelli et Piero della Francesca en sont des noms connus.",
        ),
        "en": (
            "The Early Renaissance",
            "The Early Renaissance is the 15th century in Italy, above all in Florence. Artists rediscover Antiquity, "
            "study the human body and work out linear perspective to create a believable space. Masaccio, Fra "
            "Angelico, Botticelli and Piero della Francesca are well-known names.",
        ),
    },
    "High_Renaissance": {
        "fr": (
            "La haute Renaissance",
            "La haute Renaissance désigne le début du XVIe siècle en Italie, à Florence puis à Rome. Léonard de Vinci, "
            "Michel-Ange et Raphaël y recherchent l'équilibre, l'harmonie et des figures idéalisées dans des "
            "compositions claires et monumentales. Regardez les groupes équilibrés, souvent en forme de pyramide.",
        ),
        "en": (
            "The High Renaissance",
            "The High Renaissance is the early 16th century in Italy, in Florence and then Rome. Leonardo da Vinci, "
            "Michelangelo and Raphael seek balance, harmony and idealised figures in clear, monumental compositions. "
            "Look at the balanced groups, often arranged as a pyramid.",
        ),
    },
    "Mannerism_Late_Renaissance": {
        "fr": (
            "Le maniérisme",
            "Le maniérisme se développe en Italie au XVIe siècle, après la haute Renaissance. Il préfère l'élégance et "
            "l'artifice à l'équilibre : figures étirées, poses tordues, couleurs acides. Pontormo et Parmigianino en "
            "sont des exemples, et l'on y rattache souvent le Greco. Regardez les proportions volontairement exagérées.",
        ),
        "en": (
            "Mannerism",
            "Mannerism developed in 16th-century Italy, after the High Renaissance. It prefers elegance and artifice "
            "to balance: stretched figures, twisting poses, acid colours. Pontormo and Parmigianino are examples, and "
            "El Greco is often linked to it. Look at the deliberately exaggerated proportions.",
        ),
    },
    "Rococo": {
        "fr": (
            "Le rococo",
            "Le rococo se développe dans la France du XVIIIe siècle, puis gagne l'Europe. C'est un art léger et "
            "raffiné, aux couleurs claires, qui montre les plaisirs de la vie aristocratique : fêtes galantes, amours, "
            "jardins. Watteau, Boucher et Fragonard sont les noms à retenir. Regardez la grâce et les teintes pastel.",
        ),
        "en": (
            "Rococo",
            "Rococo developed in 18th-century France, then spread across Europe. It is a light, refined art in soft "
            "colours that shows the pleasures of aristocratic life: courtship parties, love scenes, gardens. Watteau, "
            "Boucher and Fragonard are the names to remember. Look at the grace and the pastel tones.",
        ),
    },
    "Symbolism": {
        "fr": (
            "Le symbolisme",
            "Le symbolisme se développe à la fin du XIXe siècle, d'abord en France et en Belgique, en réaction au "
            "réalisme et à l'impressionnisme. Ces artistes préfèrent le rêve, le mythe et le mystère à la description "
            "du réel et cherchent à suggérer des idées par des images. Gustave Moreau et Odilon Redon en sont des "
            "figures marquantes.",
        ),
        "en": (
            "Symbolism",
            "Symbolism developed in the late 19th century, first in France and Belgium, as a reaction against Realism "
            "and Impressionism. These artists prefer dream, myth and mystery to the description of reality and try to "
            "suggest ideas through images. Gustave Moreau and Odilon Redon are leading figures.",
        ),
    },
    "Art_Nouveau": {
        "fr": (
            "L'Art nouveau",
            "L'Art nouveau fleurit en Europe vers 1890-1910, dans l'architecture, les affiches et les objets autant que "
            "dans la peinture. Il s'inspire des plantes et des fleurs et aime les lignes souples, les courbes et "
            "l'ornement. Alphonse Mucha et Gustav Klimt y sont souvent associés.",
        ),
        "en": (
            "Art Nouveau",
            "Art Nouveau flourished in Europe around 1890-1910, in architecture, posters and objects as much as in "
            "painting. It draws on plants and flowers and loves flowing lines, curves and ornament. Alphonse Mucha "
            "and Gustav Klimt are often associated with it.",
        ),
    },
    "Naive_Art_Primitivism": {
        "fr": (
            "L'art naïf",
            "L'art naïf désigne des peintres souvent autodidactes, qui ignorent ou écartent les règles académiques de "
            "la perspective et du modelé. Les formes sont simples, les couleurs franches et les détails minutieux. "
            "Henri Rousseau, dit le Douanier, en est l'exemple le plus connu. Regardez l'espace volontairement aplati.",
        ),
        "en": (
            "Naive art",
            "Naive art names painters who are often self-taught and who ignore or set aside the academic rules of "
            "perspective and modelling. Forms are simple, colours frank and details painstaking. Henri Rousseau, "
            "known as Le Douanier, is the best-known example. Look at the deliberately flattened space.",
        ),
    },
    "Fauvism": {
        "fr": (
            "Le fauvisme",
            "Le fauvisme éclate à Paris en 1905, au Salon d'automne : un critique qualifie ces peintres de « fauves » "
            "devant leurs couleurs violentes. Matisse et Derain en sont les chefs de file. La couleur pure, en larges "
            "aplats, n'imite plus la nature mais exprime la sensation. Le mouvement est bref.",
        ),
        "en": (
            "Fauvism",
            "Fauvism burst out in Paris in 1905, at the Salon d'Automne: a critic called these painters \"fauves\" "
            "(wild beasts) because of their violent colours. Matisse and Derain led the group. Pure colour, in broad "
            "flat areas, no longer imitates nature but expresses sensation. The movement was short-lived.",
        ),
    },
    "Pop_Art": {
        "fr": (
            "Le pop art",
            "Le pop art naît dans les années 1950 en Grande-Bretagne, puis s'épanouit aux États-Unis dans les années "
            "1960. Il emprunte ses images à la publicité, à la bande dessinée et à la société de consommation. Andy "
            "Warhol et Roy Lichtenstein en sont les figures les plus connues. Regardez les couleurs vives et la "
            "répétition des images.",
        ),
        "en": (
            "Pop Art",
            "Pop Art began in the 1950s in Britain, then flourished in the United States in the 1960s. It borrows its "
            "images from advertising, comics and consumer society. Andy Warhol and Roy Lichtenstein are the "
            "best-known figures. Look at the bright colours and the repetition of images.",
        ),
    },
    "Ukiyo_e": {
        "fr": (
            "L'ukiyo-e",
            "L'ukiyo-e, « images du monde flottant », est un art de l'estampe sur bois du Japon de l'époque d'Edo "
            "(1603-1868). Il montre des acteurs de kabuki, des beautés et des paysages. Hokusai et Hiroshige en sont "
            "des maîtres, et ces estampes ont influencé des artistes européens comme Monet et Van Gogh.",
        ),
        "en": (
            "Ukiyo-e",
            "Ukiyo-e, \"pictures of the floating world\", is the woodblock print art of Japan's Edo period "
            "(1603-1868). It shows kabuki actors, beauties and landscapes. Hokusai and Hiroshige are among its "
            "masters, and these prints influenced European artists such as Monet and Van Gogh.",
        ),
    },
}

GENRE_LABELS = tuple(_GENRES)
STYLE_LABELS = tuple(_STYLES)
_NOTICES = {**_GENRES, **_STYLES}

_GENERIC = {
    "fr": ("Cette tradition", "Aucune notice n'est encore disponible pour cette tradition."),
    "en": ("This tradition", "No notice is available yet for this tradition."),
}


def _key(label: str) -> str:
    """Lower-case, with every run of non-alphanumeric characters turned into one underscore."""
    return re.sub(r"[^0-9a-z]+", "_", label.strip().lower()).strip("_")


_BY_KEY = {_key(label): label for label in _NOTICES}


def normalise_label(label: str) -> str | None:
    """The canonical label for a spelling such as 'post-impressionism', 'Religious painting' or 'Ukiyo-e'."""
    return _BY_KEY.get(_key(label))


def covered_labels() -> tuple[str, ...]:
    """Every label that has a notice: the genres first, then the styles."""
    return GENRE_LABELS + STYLE_LABELS


def _lang(lang: str) -> str:
    if lang not in LANGS:
        raise ValueError(f"unknown language {lang!r}: choose one of {LANGS}")
    return lang


def title_for(label: str, lang: str = "fr", fallback: bool = False) -> str | None:
    """The tradition's name in `lang` ('Le portrait', 'Impressionism'), or None if the label is unknown."""
    lang = _lang(lang)
    canonical = normalise_label(label)
    if canonical is None:
        return _GENERIC[lang][0] if fallback else None
    return _NOTICES[canonical][lang][0]


def notice_for(label: str, lang: str = "fr", fallback: bool = False) -> str | None:
    """The notice text for a genre or style, in `lang` ('fr' or 'en').

    Unknown labels give None, or a generic sentence with `fallback=True`.
    """
    lang = _lang(lang)
    canonical = normalise_label(label)
    if canonical is None:
        return _GENERIC[lang][1] if fallback else None
    return _NOTICES[canonical][lang][1]


def story_entry(label: str, lang: str = "fr") -> dict[str, str] | None:
    """The notice as a JSON-ready dict (label, language, title, text), or None for an unknown label."""
    text = notice_for(label, lang)
    if text is None:
        return None
    return {"label": normalise_label(label), "lang": lang, "title": title_for(label, lang), "text": text}
