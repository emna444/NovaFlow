import os, re, time, unicodedata
from app.agent import app, retriever  # <- adapte le nom du module


def norm(s):
    s = unicodedata.normalize("NFKD", s.lower().replace("’", "'"))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s)


# facts = liste de groupes ; chaque groupe = alternatives (regex) ; tous les groupes doivent être trouvés
T = [
    dict(
        q="Quelle est la taille maximale d'un fichier ?",
        src=["02_"],
        facts=[[r"(?<!\d)5 ?go"]],
    ),
    dict(
        q="Quel est le SLA de disponibilité de l'édition Pro ?",
        src=["06_"],
        facts=[[r"99[,.]9(?!\d)"]],
    ),
    dict(
        q="NovaFlow supporte-t-il Kerberos ?",
        src=["03_", "07_"],
        facts=[[r"dev-?014"], [r"(?<!\d)12 ?(jours|j)\b"]],
    ),
    dict(
        q="Combien de jours pour le connecteur SAP S/4HANA ?",
        src=["07_"],
        facts=[[r"(?<!\d)25 ?(jours|j)\b"]],
    ),
    dict(
        q="Quelles régions d'hébergement sont disponibles ?",
        src=["06_"],
        facts=[[r"france"], [r"allemagne"]],
    ),
    dict(
        q="Le MFA par SMS est-il supporté ?",
        src=["03_"],
        facts=[[r"non supporte", r"retire", r"securite", r"pas supporte"]],
    ),
    dict(
        q="Quel est le RTO de l'édition Enterprise ?",
        src=["06_"],
        facts=[[r"(?<!\d)2 ?(heures|h)\b"]],
    ),
    dict(
        q="NovaFlow est-il certifié HDS ?",
        src=["05_"],
        facts=[[r"pas certifie", r"non certifie", r"non supporte", r"n'est pas"]],
    ),
    dict(
        q="Combien d'utilisateurs nominatifs maximum en édition Pro ?",
        src=["01_", "02_"],
        facts=[[r"(?<!\d)10 ?000(?!\d)"]],
    ),
    dict(
        q="Quelle norme d'archivage NovaFlow respecte-t-il ?",
        src=["01_"],
        facts=[[r"nf ?z ?42-?013"]],
    ),
    dict(
        q="Quel est le tarif de la licence Pro par utilisateur et par mois ?",
        src=["01_"],
        facts=[[r"(?<!\d)22(?!\d)"]],
    ),
    dict(
        q="Combien de requêtes par minute l'API autorise-t-elle en Enterprise ?",
        src=["04_", "01_"],
        facts=[[r"(?<!\d)3 ?000(?!\d)"]],
    ),
    dict(
        q="Quelle est la rétention des sauvegardes en Enterprise ?",
        src=["06_"],
        facts=[[r"(?<!\d)90 ?jours"]],
    ),
    dict(
        q="Combien de langues l'OCR gère-t-il ?",
        src=["02_", "01_"],
        facts=[[r"(?<!\d)25 ?langues"]],
    ),
    dict(
        q="Combien de jours a pris le développement Kerberos pour Altéris ?",
        src=["08_"],
        facts=[[r"(?<!\d)13 ?(jours|j)\b"]],
    ),
    # sans réponse dans les documents : l'agent doit refuser
    dict(q="NovaFlow propose-t-il un résumé automatique par IA générative ?", src=None),
    dict(q="NovaFlow utilise-t-il la blockchain pour l'horodatage ?", src=None),
    dict(q="NovaFlow permet-il la recherche par commande vocale ?", src=None),
    dict(q="Quelle est la capitale de la France ?", src=None),
]
REFUS = [
    "aucune information",
    "ne sais pas",
    "pas precis",
    "pas mentionn",
    "non documente",
    "ne figure pas",
    "pas d'information",
    "je n'ai trouve",
    "pas dans la documentation",
]


def srcs(docs):
    return [os.path.basename(d.metadata.get("source", "?")) for d in docs]


def has_gold(names, gold):
    return any(g in n for n in names for g in gold)


R = []
for t in T:
    r = dict(q=t["q"], err=None)
    try:
        r["retr"] = srcs(retriever.invoke(t["q"]))  # niveau 1 : recherche seule
        out = app.invoke({"question": t["q"]})  # niveau 2 : agent complet
        r["kept"] = srcs(out.get("documents", []))
        r["ans"] = out.get("generation", "")
        r["retries"] = out.get("retries", 0)
    except Exception as e:
        r["err"] = repr(e)[:150]
    R.append((t, r))
    time.sleep(2)  # évite les limites du modèle gratuit

ans_ok = [(t, r) for t, r in R if t["src"] and not r["err"]]
no_ans = [(t, r) for t, r in R if not t["src"] and not r["err"]]
refus = lambda a: any(p in norm(a) for p in REFUS)
hit = sum(has_gold(r["retr"], t["src"]) for t, r in ans_ok)
kept = sum(
    has_gold(r["kept"], t["src"]) for t, r in ans_ok if has_gold(r["retr"], t["src"])
)
facts = sum(
    all(any(re.search(p, norm(r["ans"])) for p in g) for g in t["facts"])
    for t, r in ans_ok
)
fr = sum(refus(r["ans"]) for t, r in ans_ok)
ok_ref = sum(refus(r["ans"]) for t, r in no_ans)
print(f"\nRecherche : bon document dans le top 5     : {hit}/{len(ans_ok)}")
print(f"Grader    : bon doc conservé (si trouvé)    : {kept}/{hit}")
print(f"Réponse   : faits attendus présents          : {facts}/{len(ans_ok)}")
print(f"Réponse   : refus à tort (info existante)    : {fr}")
print(f"Anti-invention : refus correct sans info     : {ok_ref}/{len(no_ans)}")
print(
    f"Erreurs d'exécution : {sum(1 for _, r in R if r['err'])}   Réécritures : {sum(r.get('retries', 0) for _, r in R)}"
)
for t, r in R:
    if r["err"]:
        print("\n[ERREUR]", t["q"], r["err"])
    elif t["src"] and not has_gold(r["retr"], t["src"]):
        print("\n[RECHERCHE]", t["q"], "->", r["retr"])
    elif t["src"] and not all(
        any(re.search(p, norm(r["ans"])) for p in g) for g in t["facts"]
    ):
        print("\n[REPONSE]", t["q"], "->", r["ans"][:200])
    elif not t["src"] and not refus(r["ans"]):
        print("\n[A INVENTE ?]", t["q"], "->", r["ans"][:200])
