"""A small web page for the guide: send a photo (or use the webcam), ask a question, read the answer.

    pip install -e ".[demo]"
    python scripts/demo.py                  # then open http://127.0.0.1:7860
    python scripts/demo.py --lang en --port 8000
    python scripts/demo.py --offline         # no connection at all (CLIP's weights must be in the local cache)

The page says whether the photo is a painting, its most likely genre (or "not sure"), the story of that
tradition, similar paintings of the collection and, if you ask a question, the paintings that answer it.
The computation happens on this machine and the server listens on 127.0.0.1 only unless you pass --host. Gradio's
analytics are switched off. Loading CLIP, however, contacts the Hugging Face hub (to download the weights the first time
and to check the cached files afterwards): pass --offline once the weights are cached to forbid any such connection.
The first start also builds the collection index if `data/search_index.npz` does not exist (about 30 s).

This is a demonstration, not a kiosk application. The gallery shows WikiArt paintings from your local `data/`
folder, which must not be published, and the story texts are drafts written by an AI, pending review.
"""
import argparse
import os
import sys

# Switch the telemetry of the web libraries off BEFORE importing them.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from museum_guide.demo_logic import STATUSES, UI_TEXT, DemoGuide, display, text  # noqa: E402
from museum_guide.notices import LANGS  # noqa: E402

BANNER_COLOURS = {  # background, text: both set, so the banner reads the same in light and dark themes
    "painting": ("#d9f2e3", "#14532d"),
    "unsure": ("#fdf0c8", "#713f12"),
    "not_a_painting": ("#f9d6d6", "#7f1d1d"),
    "no_image": ("#e5e7eb", "#374151"),
}
assert set(BANNER_COLOURS) == set(STATUSES)


def banner(kind: str, message: str) -> str:
    """The coloured verdict banner as HTML (the message is escaped: it may contain a genre name)."""
    import html

    background, colour = BANNER_COLOURS[kind]
    return (
        f'<div style="background:{background};color:{colour};padding:14px 18px;border-radius:10px;'
        f'font-size:1.15rem;font-weight:600">{html.escape(message)}</div>'
    )


def gallery(items, label=None):
    """A gallery's new state: its pictures (and label), hidden while it has none (an empty box would take 320 px)."""
    changes = {"value": items, "visible": bool(items)}
    if label is not None:
        changes["label"] = label
    return import_gradio().update(**changes)


def import_gradio():
    try:
        import gradio as gr
    except ImportError:
        raise SystemExit('Gradio is not installed: run `pip install -e ".[demo]"`.') from None
    return gr


def build_demo(guide: DemoGuide, lang: str = "fr"):
    """The Gradio page for a loaded `DemoGuide`. Nothing is computed until a visitor sends something."""
    gr = import_gradio()

    def render(result, page_lang):
        """The values of the output components for a result, in the page's language."""
        d = display(result, guide.data_dir, page_lang)
        story = f"### {d.story_title}\n\n{d.story_text}\n\n*{d.story_note}*" if d.story_text else ""
        return banner(d.kind, d.verdict), d.genres or None, story, d.similar, d.answers

    def respond(photo, question, page_lang):
        result = guide.answer(photo, question, page_lang)
        verdict_html, genre_values, story_md, similar_items, answer_items = render(result, page_lang)
        return verdict_html, genre_values, story_md, gallery(similar_items), gallery(answer_items), result

    def switch_language(page_lang, result):
        """Re-label the page and re-render the last result in the new language, without encoding anything again."""
        t = UI_TEXT[page_lang]
        if result is not None:
            verdict_html, genre_values, story_md, similar_items, answer_items = render(result, page_lang)
        else:
            verdict_html, genre_values, story_md, similar_items, answer_items = banner("no_image", t["no_image"]), None, "", [], []
        return (
            verdict_html,
            story_md,
            gr.update(value=genre_values, label=t["genres"]),
            gallery(similar_items, t["similar"]),
            gallery(answer_items, t["answers"]),
            gr.update(value=f"# {t['title']}\n\n{t['intro']}"),
            gr.update(label=t["photo"]),
            gr.update(label=t["question"], placeholder=t["question_placeholder"]),
            gr.update(label=t["language"]),
            gr.update(value=t["submit"]),
        )

    t = UI_TEXT[lang]
    with gr.Blocks(title=t["title"], analytics_enabled=False) as demo:
        intro = gr.Markdown(f"# {t['title']}\n\n{t['intro']}")
        last = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                photo = gr.Image(label=t["photo"], type="pil", sources=["upload", "webcam"], height=360)
                question = gr.Textbox(label=t["question"], placeholder=t["question_placeholder"], lines=1, max_lines=2)
                language = gr.Radio([("Français", "fr"), ("English", "en")], value=lang, label=t["language"])
                submit = gr.Button(t["submit"], variant="primary")
            with gr.Column(scale=1):
                verdict = gr.HTML(banner("no_image", t["no_image"]))
                genres = gr.Label(label=t["genres"], num_top_classes=3)
                story = gr.Markdown()
        similar = gr.Gallery(label=t["similar"], columns=3, height=320, object_fit="contain", allow_preview=True, visible=False)
        answers = gr.Gallery(label=t["answers"], columns=3, height=320, object_fit="contain", allow_preview=True, visible=False)

        outputs = [verdict, genres, story, similar, answers, last]
        gr.on(triggers=[submit.click, question.submit, photo.change], fn=respond, inputs=[photo, question, language], outputs=outputs)
        language.change(
            switch_language, inputs=[language, last],
            outputs=[verdict, story, genres, similar, answers, intro, photo, question, language, submit],
        )
    demo.queue(default_concurrency_limit=1)  # one CLIP in memory: one request at a time
    return demo


def main(argv=None, guide=None) -> int:
    """`guide` replaces the loaded `DemoGuide` (tests)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="127.0.0.1", help="address to listen on (default: this machine only)")
    p.add_argument("--lang", choices=LANGS, default="fr", help="language of the page at start (switchable on the page)")
    p.add_argument("--share", action="store_true", help="ask Gradio for a public link (off by default: it exposes the demo to the internet)")
    p.add_argument("--k", type=int, default=5, help="paintings shown per gallery")
    p.add_argument("--offline", action="store_true", help="never contact the Hugging Face hub (needs the weights in the local cache)")
    args = p.parse_args(argv)
    if args.offline:  # before anything imports the hub client, which reads these once
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import_gradio()  # fail early, with the install command, before loading CLIP
    try:
        guide = guide or DemoGuide.load(k=args.k)
    except (OSError, ValueError) as err:
        print(err, file=sys.stderr)
        return 1
    if not guide.rejects_non_paintings:
        print("note: reports/non_painting_rule.json not found, so 'not a painting' cannot be said "
              "(fit it with `python scripts/non_painting.py`).", file=sys.stderr)
    demo = build_demo(guide, args.lang)
    print(f"{text(args.lang, 'title')}: http://{args.host}:{args.port}")
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, show_error=True, ssr_mode=False,
                footer_links=[], quiet=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
