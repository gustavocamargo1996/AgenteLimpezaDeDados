"""Mede densidade de comentario por arquivo. Nao e' teste; e' instrumento."""
import ast
import glob
import io
import tokenize


def medir(caminho: str) -> dict:
    src = open(caminho, encoding="utf-8").read()
    total = len(src.splitlines())
    doc = 0
    for no in ast.walk(ast.parse(src)):
        if isinstance(no, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            d = ast.get_docstring(no, clean=False)
            if d:
                doc += len(d.splitlines()) + 2
    com = sum(1 for t in tokenize.generate_tokens(io.StringIO(src).readline)
              if t.type == tokenize.COMMENT)
    return {"arquivo": caminho, "total": total, "doc": doc, "comentario": com,
            "densidade": (doc + com) / total if total else 0}


if __name__ == "__main__":
    alvos = ["main.py", "avaliar_limpador.py"] + glob.glob("limpeza/**/*.py", recursive=True)
    linhas = [medir(a) for a in alvos]
    for m in sorted(linhas, key=lambda x: -x["densidade"]):
        print(f"{m['arquivo']:40s}{m['total']:6d}{m['densidade']:8.1%}")
    t = sum(m["total"] for m in linhas)
    d = sum(m["doc"] + m["comentario"] for m in linhas)
    print(f"\nTOTAL {t} linhas, {d/t:.1%} de comentario")
