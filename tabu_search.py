# tabu_search.py
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

def gerar_vizinho(alocacao, custo, oferta, demanda):
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

def busca_tabu_transporte(custo, oferta, demanda, max_iter=100, tamanho_tabu=5):
    """
    Aplica a metaheurística da Busca Tabu para minimizar o custo
    do problema de transporte.
    
    Parâmetros:
      - custo: Matriz de custos.
      - oferta: Vetor de oferta.
      - demanda: Vetor de demanda.
      - max_iter: Número máximo de iterações.
      - tamanho_tabu: Tamanho máximo da lista tabu (para evitar movimentos repetitivos).
      
    Retorna:
      - melhor_solucao: Matriz de alocação com o menor custo encontrado.
      - melhor_custo: Custo total da melhor solução.
      - registro_iteracoes: Histórico das iterações (iteração, custo, alocação, tempo decorrido).
    """
    
    tempo_inicial = time.perf_counter_ns()

    total_oferta = oferta.sum()
    total_demanda = demanda.sum()
    matriz_custo = custo.copy()
    vetor_oferta = oferta.copy()
    vetor_demanda = demanda.copy()

    # Balanceamento: adiciona linha ou coluna dummy se a oferta e demanda não forem iguais
    if total_oferta > total_demanda:
        custo_dummy = np.zeros((matriz_custo.shape[0], 1))
        matriz_custo = np.hstack([matriz_custo, custo_dummy])
        vetor_demanda = np.append(vetor_demanda, total_oferta - total_demanda)
    elif total_demanda > total_oferta:
        custo_dummy = np.zeros((1, matriz_custo.shape[1]))
        matriz_custo = np.vstack([matriz_custo, custo_dummy])
        vetor_oferta = np.append(vetor_oferta, total_demanda - total_oferta)

    # Gera a solução inicial pelo critério do menor custo
    solucao_atual = solucao_inicial_menor_custo(matriz_custo, vetor_oferta, vetor_demanda)  # Corrigido nome da função
    melhor_solucao = solucao_atual.copy()
    melhor_custo = calcular_custo_total(matriz_custo, melhor_solucao)

    registro_iteracoes = []
    lista_tabu = []
    
    # Loop principal da Busca Tabu  
    for iteracao in range(max_iter):
        vizinhos = []
        for _ in range(10):   # Gera 10 soluções vizinhas aleatórias
            vizinho = gerar_vizinho(solucao_atual, matriz_custo, vetor_oferta, vetor_demanda)
            vizinhos.append(vizinho)
            
         # Calcula o custo total para cada vizinho
        custos_vizinhos = [calcular_custo_total(matriz_custo, vz) for vz in vizinhos]

        melhor_vizinho = None
        melhor_custo_vizinho = float('inf')
        melhor_movimento = None
        
        # Seleciona o melhor vizinho que não esteja na lista tabu
        for vz, vz_custo in zip(vizinhos, custos_vizinhos):
            # Define o movimento como a soma das diferenças absolutas entre o vizinho e a solução atual
            movimento = np.sum(np.abs(vz - solucao_atual))
            if movimento in lista_tabu:
                continue
            if vz_custo < melhor_custo_vizinho:
                melhor_custo_vizinho = vz_custo
                melhor_vizinho = vz
                melhor_movimento = movimento

        # Se todos os movimentos estiverem na lista tabu, seleciona o primeiro vizinho
        if melhor_vizinho is None:
            melhor_vizinho = vizinhos[0]
            melhor_custo_vizinho = custos_vizinhos[0]
            melhor_movimento = np.sum(np.abs(melhor_vizinho - solucao_atual))
            
        # Atualiza a solução atual para o melhor vizinho encontrado
        solucao_atual = melhor_vizinho

        # Atualiza a melhor solução se o novo vizinho tiver custo inferior
        if melhor_custo_vizinho < melhor_custo:
            melhor_solucao = melhor_vizinho.copy()
            melhor_custo = melhor_custo_vizinho

       # Adiciona o movimento realizado à lista tabu e controla seu tamanho
        lista_tabu.append(melhor_movimento)
        if len(lista_tabu) > tamanho_tabu:
            lista_tabu.pop(0)

        elapsed = time.perf_counter_ns() - tempo_inicial
        elapsed = elapsed / 1000000 #time control  # Converte o tempo para milissegundos
        # Registra os dados da iteração (número, custo, solução, tempo decorrido)
        registro_iteracoes.append((iteracao, melhor_custo_vizinho, solucao_atual.copy(), elapsed))

     # Remove a linha ou coluna dummy, se adicionada, para retornar à estrutura original
    if total_oferta > total_demanda:
        melhor_solucao = melhor_solucao[:, :-1]
    elif total_demanda > total_oferta:
        melhor_solucao = melhor_solucao[:-1, :]

    melhor_custo = calcular_custo_total(custo, melhor_solucao)
    return melhor_solucao, melhor_custo, registro_iteracoes
