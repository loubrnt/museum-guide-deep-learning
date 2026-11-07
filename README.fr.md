# Guide de musée : reconnaissance d'œuvres par apprentissage profond (résumé en français)

Projet final du cours **Neural Networks & Deep Learning Foundations** (MSA-DATB09-91, Albert School, 2026-27).
**Groupe 5 :** Lou Brunet, Mathis Meimoun, Alexis Maeder.

> Ceci est un résumé d'une page. La documentation complète, avec toutes les mesures, est en anglais dans le [README](README.md).

## Le sujet

Un musée français des beaux-arts (fictif) veut un guide intelligent pour les téléphones des visiteurs et les bornes des salles :

1. **Reconnaître** le tableau que le visiteur photographie : son genre ou son style, et raconter l'histoire de cette tradition.
2. **Répondre à des questions** en langage naturel (« Quel tableau montre une mer agitée ? ») avec les œuvres qui correspondent le mieux.

Le directeur demande aussi une recommandation : les bornes ont des processeurs modestes, l'équipe est petite et le budget limité.
Un réseau de neurones vaut-il le coup en production, ou un modèle classique plus simple suffirait-il ?

## La réponse

**Oui, un réseau de neurones vaut le coup, à condition qu'il soit pré-entraîné.** Entraîné de zéro sur quelques centaines d'images, il fait moins bien que le modèle classique. Le modèle proposé est **l'encodeur d'image de CLIP avec une simple tête linéaire**. Le ResNet-18 affiné sert de solution de repli si la borne est très limitée en mémoire ou en puissance.

Précision sur le jeu de test, séparé **par artiste** (aucun peintre nommé n'apparaît à la fois dans l'entraînement et dans le test), moyenne sur 10 graines :

| Approche | Genre (5 classes) | Style (6 classes) |
|---|---|---|
| Régression logistique (couleurs et contours) | 49,0 % ± 3,7 | 32,1 % ± 2,2 |
| Forêt aléatoire (couleurs et contours) | 53,9 % ± 5,2 | 32,4 % ± 2,2 |
| Réseau dense entraîné de zéro | 40,9 % ± 6,0 | non mesuré |
| ResNet-18 pré-entraîné et affiné | 74,0 % ± 3,8 | 40,6 % ± 5,4 |
| **CLIP (encodeur d'image) + tête linéaire** | **90,2 % ± 2,5** | **55,4 % ± 5,2** |

Le hasard est à 20 % pour le genre et à 16,7 % pour le style.

**Ce que ça coûte sur une borne** (Apple M4, un seul cœur, lecture de l'image comprise) :

| Modèle | Fichier de poids | Temps par image | Mémoire ajoutée |
|---|---|---|---|
| ResNet-18 | 44,8 Mo | 7,2 ms | 120 Mo |
| CLIP, encodeur d'image | 351,5 Mo | 20,9 ms | 729 Mo |

Un processeur de borne sera plus lent que ce portable : les rapports comptent plus que les millisecondes.

## Ce que le projet contient en plus

- Un **guide** qui reconnaît le genre, affiche une courte notice sur la tradition, propose des tableaux proches et répond à une question, avec une confiance calibrée, une réponse « pas sûr » et le refus de ce qui n'est pas un tableau.
- Une **recherche** par question en langage naturel, y compris en français (les noms d'artistes demandent une attention particulière : l'orthographe française change les résultats).
- Des mesures de **robustesse** aux conditions de prise de vue : face à une rotation, un flou ou une perspective, la tête CLIP garde 82,5 % en moyenne sur neuf perturbations, contre 64,3 % pour le ResNet (5 graines).
- Un **notebook** qui raconte le projet en graphiques ([`notebooks/museum_guide_walkthrough.ipynb`](notebooks/museum_guide_walkthrough.ipynb)), une **démo web**, des tests, et un [historique des modifications](CHANGELOG.md).

## Pour essayer

```bash
pip install -e ".[dev]"          # le paquet et les outils de test
make test                        # données synthétiques : rien à télécharger
make data                        # construit le sous-ensemble de WikiArt (environ 2,3 Go)
python scripts/guide.py photo.jpg
```

## Les limites, dites franchement

- **Aucune vraie photo n'a encore été testée.** Les mesures de robustesse utilisent des perturbations synthétiques des images du jeu de données. L'outil et le protocole pour tester de vraies photos sont prêts ([`docs/REAL_PHOTO_PROTOCOL.md`](docs/REAL_PHOTO_PROTOCOL.md)).
- Seulement **5 genres et 6 styles**, sur 1 500 à 1 800 images de WikiArt : bien moins que les collections d'un musée.
- Les **notices sur les traditions** et les **traductions françaises** des questions ont été rédigées avec l'aide d'une IA : elles doivent être relues avant d'être montrées à des visiteurs.
- Les images de WikiArt ont une licence « inconnue » : elles ne sont pas redistribuées dans le dépôt, mais quelques-unes restent visibles dans l'historique Git et dans la présentation. Aucune licence n'a encore été choisie pour le code.
