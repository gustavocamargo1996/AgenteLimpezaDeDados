"""POC: agente gerador de cadeias de pensamento para regras de correcao.

Inspirada no ZeroDC (github.com/YangChen32768/ZeroDC), simplificada e com dois
desvios deliberados: o KMeans roda sobre valores distintos em vez de celulas, e
o codigo gerado passa por portao AST + namespace restrito antes de executar.
"""
__version__ = "0.1.0"
