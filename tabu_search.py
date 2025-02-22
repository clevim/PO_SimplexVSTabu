import numpy as np
import random
import time
from simplex import solucao_inicial_menor_custo

def calcular_custo_total(custo, alocacao):
    """
    Calcula o custo total da solução multiplicando, elemento a elemento,
    a matriz de alocação pela matriz de custos.
    
    Parâmetros:
      - custo: Matriz de custos.
      - alocacao: Matriz de alocação.
      
    Retorna:
      - Custo total (soma dos produtos).
    """
    return np.sum(alocacao * custo)

def gerar_vizinho(alocacao, custo, oferta, demanda , num_swaps=3):
    """
    Gera uma solução vizinha a partir da solução atual, alterando
    aleatoriamente o valor de uma célula na matriz de alocação.
    
    Parâmetros:
      - alocacao: Matriz de alocação atual.
      - custo: Matriz de custos (não utilizada diretamente para a modificação).
      - oferta, demanda: Vetores de oferta e demanda (mantidos inalterados).
      
    Retorna:
      - nova_alocacao: Nova matriz de alocação com alteração em uma célula.
    """
    m, n = alocacao.shape
    nova_alocacao = alocacao.copy()
    # Seleciona aleatoriamente uma célula da matriz
    i = random.randint(0, m - 1)
    j = random.randint(0, n - 1)
    # Escolhe aleatoriamente adicionar ou subtrair 1, garantindo que o valor não fique negativo
    delta = random.choice([-1, 1])
    nova_alocacao[i, j] = max(nova_alocacao[i, j] + delta, 0)
    return nova_alocacao

def gerar_vizinhos(alocacao, num_swaps=3, num_vizinhos=10):
    """
    Gera uma lista de vizinhos realizando múltiplas trocas entre células.
    """
    m, n = alocacao.shape
    vizinhos = []
    
    for _ in range(num_vizinhos):
        nova_alocacao = alocacao.copy()

        for _ in range(num_swaps):
            i1, j1 = random.randint(0, m - 1), random.randint(0, n - 1)
            i2, j2 = random.randint(0, m - 1), random.randint(0, n - 1)
            while (i1 == i2 and j1 == j2):
                i2, j2 = random.randint(0, m - 1), random.randint(0, n - 1)
            nova_alocacao[i1, j1], nova_alocacao[i2, j2] = nova_alocacao[i2, j2], nova_alocacao[i1, j1]
        
        vizinhos.append(nova_alocacao)
    
    return vizinhos

def busca_tabu_transporte(custo, oferta, demanda, max_iter=100, tamanho_tabu=5):
    tempo_inicial = time.perf_counter_ns()  # Medição em nanossegundos

    total_oferta = oferta.sum()
    total_demanda = demanda.sum()
    matriz_custo = custo.copy()
    vetor_oferta = oferta.copy()
    vetor_demanda = demanda.copy()

    if total_oferta > total_demanda:
        custo_dummy = np.zeros((matriz_custo.shape[0], 1))
        matriz_custo = np.hstack([matriz_custo, custo_dummy])
        vetor_demanda = np.append(vetor_demanda, total_oferta - total_demanda)
    elif total_demanda > total_oferta:
        custo_dummy = np.zeros((1, matriz_custo.shape[1]))
        matriz_custo = np.vstack([matriz_custo, custo_dummy])
        vetor_oferta = np.append(vetor_oferta, total_demanda - total_oferta)

    solucao_atual = solucao_inicial_menor_custo(matriz_custo, vetor_oferta, vetor_demanda)
    melhor_solucao = solucao_atual.copy()
    melhor_custo = calcular_custo_total(matriz_custo, melhor_solucao)

    registro_iteracoes = []
    lista_tabu = []
    registro_iteracoes.append((0, melhor_custo, solucao_atual.copy(), 0))

    for iteracao in range(1,max_iter):
        vizinhos = gerar_vizinhos(solucao_atual)
        custos_vizinhos = [calcular_custo_total(matriz_custo, vz) for vz in vizinhos]

        melhor_vizinho = None
        melhor_custo_vizinho = float('inf')
        melhor_movimento = None
        
        for vz, vz_custo in zip(vizinhos, custos_vizinhos):
            movimento = np.sum(np.abs(vz - solucao_atual))
            if movimento in lista_tabu:
                continue
            if vz_custo < melhor_custo_vizinho:
                melhor_custo_vizinho = vz_custo
                melhor_vizinho = vz
                melhor_movimento = movimento

        if melhor_vizinho is None:
            melhor_vizinho = vizinhos[0]
            melhor_custo_vizinho = custos_vizinhos[0]
            melhor_movimento = np.sum(np.abs(melhor_vizinho - solucao_atual))

        solucao_atual = melhor_vizinho

        if melhor_custo_vizinho < melhor_custo:
            melhor_solucao = melhor_vizinho.copy()
            melhor_custo = melhor_custo_vizinho

        lista_tabu.append(melhor_movimento)
        if len(lista_tabu) > tamanho_tabu:
            lista_tabu.pop(0)

        # Converter nanossegundos para microssegundos (divisão por 1000)
        elapsed = (time.perf_counter_ns() - tempo_inicial) / 1000.0
        registro_iteracoes.append((iteracao, melhor_custo_vizinho, solucao_atual.copy(), elapsed))

    if total_oferta > total_demanda:
        melhor_solucao = melhor_solucao[:, :-1]
    elif total_demanda > total_oferta:
        melhor_solucao = melhor_solucao[:-1, :]

    melhor_custo = calcular_custo_total(custo, melhor_solucao)
    return melhor_solucao, melhor_custo, registro_iteracoes
