import subprocess
import sys
import os
import numpy as np

from simplex import transporte_simplex
from tabu_search import busca_tabu_transporte
from utils import plotar_comparacao, animar_iteracoes_cenario, animar_evolucao_alocacao

def executar_cenario(nome_cenario, custo, oferta, demanda, max_iter_simplex, max_iter_tabu):
    
    """
    Executa um cenário de teste para o problema de transporte utilizando
    os métodos Simplex e Busca Tabu, e gera visualizações dos resultados.
    
    Parâmetros:
      - nome_cenario: Identificador do cenário.
      - custo: Matriz de custos.
      - oferta: Vetor de oferta.
      - demanda: Vetor de demanda.
      - max_iter_simplex: Número máximo de iterações para o método Simplex.
      - max_iter_tabu: Número máximo de iterações para a Busca Tabu.
    """
    print(f"\n=== Executando Cenário: {nome_cenario} ===")

     # Executa o método Simplex para obter a solução inicial e seu histórico
    alocacao_simp, custo_simp, registro_simp = transporte_simplex(custo, oferta, demanda, max_iter=max_iter_simplex)
    historico_custo_simp = [x[1] for x in registro_simp]
    historico_tempo_simp = [x[3] for x in registro_simp]
    if not historico_tempo_simp:
        print("Aviso: Log de tempo do Simplex está vazio!")

    # Executa a Busca Tabu para obter uma solução alternativa e seu histórico
    alocacao_tabu, custo_tabu, registro_tabu = busca_tabu_transporte(custo, oferta, demanda, max_iter=max_iter_tabu, tamanho_tabu=5)
    historico_custo_tabu = [x[1] for x in registro_tabu]
    historico_tempo_tabu = [x[3] for x in registro_tabu]

    # Exibe os resultados finais de ambos os métodos
    print(f"Simplex => Custo Final: {custo_simp:.2f} ({len(registro_simp)} iterações)")
    print("Alocação Final (Simplex):\n", alocacao_simp)
    print(f"Tabu    => Custo Final: {custo_tabu:.2f} ({len(registro_tabu)} iterações)")
    print("Alocação Final (Tabu):\n", alocacao_tabu)

    
    print(f"Dimensões Simplex: {alocacao_simp.shape}")
    print(f"Dimensões Tabu: {alocacao_tabu.shape}")

    # for value in historico_tempo_simp:
    #     print(f"v: {value}")

    # Tempo total (último registro de cada log)
    total_time_simp = historico_tempo_simp[-1] if historico_tempo_simp else 0
    total_time_tabu = historico_tempo_tabu[-1] if historico_tempo_tabu else 0

     # Geração de visualizações: gráficos comparativos e animações
    try:
        # 1) Gera as comparações estáticas (custo, tempo, heatmaps)
        plotar_comparacao(custo_simp, custo_tabu, total_time_simp, total_time_tabu,
                        alocacao_simp, alocacao_tabu, nome_cenario)
        print("Gráficos comparativos gerados com sucesso")

        # 2) Gera a animação de evolução de métricas (custo e tempo)
        animar_iteracoes_cenario(historico_custo_simp, historico_tempo_simp,
                                    historico_custo_tabu, historico_tempo_tabu,
                                    nome_cenario)
        print("Animação de métricas gerada com sucesso")

        # 3) Gera animações de alocação (heatmaps) ao longo das iterações
        animar_evolucao_alocacao(registro_simp, f"{nome_cenario}_Simplex")
        animar_evolucao_alocacao(registro_tabu, f"{nome_cenario}_Tabu")
        print("Animações de alocação geradas com sucesso")

    except Exception as e:
        print(f"Erro ao gerar visualizações: {str(e)}")

def main():
    
    # Cenário 1: Desbalanceado (oferta e demanda diferentes)
    custo1 = np.array([
        [8, 6, 10, 9],
        [9, 12, 13, 7],
        [14, 9, 16, 5]
    ], dtype=float)
    oferta1 = np.array([20, 30, 25], dtype=float)
    demanda1 = np.array([10, 25, 25, 10], dtype=float)
    executar_cenario("Desbalanceado", custo1, oferta1, demanda1, max_iter_simplex=50, max_iter_tabu=50)

    # Cenário 2: Degenerado (situação com custos uniformes)
    custo2 = np.array([
        [5, 5, 5],
        [5, 5, 5],
        [5, 5, 5]
    ], dtype=float)
    oferta2 = np.array([30, 20, 10], dtype=float)
    demanda2 = np.array([30, 20, 10], dtype=float)
    executar_cenario("Degenerado", custo2, oferta2, demanda2, max_iter_simplex=50, max_iter_tabu=50)

    # Cenário 3: Melhor desempenho da Busca Tabu
    custo3 = np.array([
        [20, 25, 15, 10, 30, 35],
        [10, 30, 20, 25, 15, 20],
        [25, 15, 30, 20, 25, 30],
        [15, 20, 25, 30, 10, 15],
        [30, 25, 20, 15, 35, 25]
    ], dtype=float)
    oferta3 = np.array([50, 60, 50, 40, 45], dtype=float)
    demanda3 = np.array([40, 45, 55, 35, 35, 35], dtype=float)
    executar_cenario("TabuMelhor", custo3, oferta3, demanda3, max_iter_simplex=10, max_iter_tabu=100)

    # Cenário 4: Melhor desempenho do método Simplex
    custo4 = np.array([
        [2,  10, 15, 20],
        [4,   8, 16, 18],
        [6,  12, 14, 10]
    ], dtype=float)
    oferta4 = np.array([40, 35, 25], dtype=float)
    demanda4 = np.array([30, 25, 25, 20], dtype=float)
    executar_cenario("SimplexMelhor", custo4, oferta4, demanda4, max_iter_simplex=50, max_iter_tabu=50)

if __name__ == "__main__":
    main()
