import numpy as np
import time

def solucao_inicial_menor_custo(custo, oferta, demanda):
    """
    Gera uma solução inicial para o problema de transporte utilizando
    o critério do menor custo.
    
    Parâmetros:
      - custo: Matriz de custos.
      - oferta: Vetor com a quantidade disponível em cada origem.
      - demanda: Vetor com a quantidade requerida em cada destino.
      
    Retorna:
      - alocacao: Matriz com as quantidades alocadas.
    """    
    
    m, n = custo.shape
    alocacao = np.zeros((m, n))
    oferta_restante = oferta.copy()
    demanda_restante = demanda.copy()
    
    # Enquanto ainda houver oferta e demanda disponíveis
    while np.any(oferta_restante > 0) and np.any(demanda_restante > 0):
        min_val = np.inf
        i_min, j_min = -1, -1
        # Procura a célula com o menor custo entre as disponíveis
        for i in range(m):
            for j in range(n):
                if oferta_restante[i] > 0 and demanda_restante[j] > 0:
                    if custo[i, j] < min_val:
                        min_val = custo[i, j]
                        i_min, j_min = i, j
        # Se não houver célula válida, encerra o laço
        if i_min == -1 or j_min == -1:
            break
         # Define a quantidade a ser alocada (mínimo entre oferta e demanda)
        aloc = min(oferta_restante[i_min], demanda_restante[j_min])
        alocacao[i_min, j_min] = aloc
        # Atualiza a oferta e a demanda remanescentes
        oferta_restante[i_min] -= aloc
        demanda_restante[j_min] -= aloc

    return alocacao

def calcular_potenciais(custo, alocacao):
    """
    Calcula os potenciais (u para linhas e v para colunas) utilizando
    o método MODI, com base na solução básica atual.
    
    Parâmetros:
      - custo: Matriz de custos.
      - alocacao: Matriz de alocação atual.
      
    Retorna:
      - u: Lista de potenciais para as linhas.
      - v: Lista de potenciais para as colunas.
    """
    m, n = custo.shape
    u = [None] * m
    v = [None] * n
    u[0] = 0   # Define o potencial da primeira linha como 0
    
     # Seleciona as células básicas (com alocação > 0)
    celulas_basicas = [(i, j) for i in range(m) for j in range(n) if alocacao[i, j] > 0]

    changed = True
    
    # Atualiza os potenciais enquanto houver alterações
    while changed:
        changed = False
        for (i, j) in celulas_basicas:
            if u[i] is not None and v[j] is None:
                v[j] = custo[i, j] - u[i]
                changed = True
            elif v[j] is not None and u[i] is None:
                u[i] = custo[i, j] - v[j]
                changed = True

    return u, v

def encontrar_variavel_entrada(custo, alocacao, u, v):
    """
    Identifica a célula não básica (alocação zero) que pode melhorar a solução,
    isto é, aquela cujo custo de oportunidade (delta) seja negativo.
    
    Parâmetros:
      - custo: Matriz de custos.
      - alocacao: Matriz de alocação atual.
      - u: Potenciais das linhas.
      - v: Potenciais das colunas.
      
    Retorna:
      - entrada: Tupla (i, j) da célula candidata.
      - min_delta: Valor do delta dessa célula.
    """
    m, n = custo.shape
    min_delta = 0
    entrada = None
    # Percorre todas as células com alocação zero
    for i in range(m):
        for j in range(n):
            if alocacao[i, j] == 0:
                # Se u[i] ou v[j] for None, delta fica "incompleto"
                if (u[i] is not None) and (v[j] is not None):
                    delta = custo[i, j] - (u[i] + v[j])
                else:
                    delta = 0  # Em casos degenerados
                if delta < min_delta:
                    min_delta = delta
                    entrada = (i, j)
    return entrada, min_delta

def encontrar_ciclo(alocacao, inicio):
    """
    Encontra um ciclo fechado (loop) que inclua a célula 'inicio'.
    Esse ciclo é necessário para ajustar a alocação e melhorar a solução.
    Usamos uma busca DFS simples para encontrar um ciclo básico.
    
    Parâmetros:
      - alocacao: Matriz de alocação atual.
      - inicio: Tupla (i, j) representando a célula candidata.
      
    Retorna:
      - ciclo: Lista de tuplas que formam o ciclo fechado.
    """
    m, n = alocacao.shape
    # Obtém as células básicas e inclui a célula de entrada provisoriamente
    basicas = [(i, j) for i in range(m) for j in range(n) if alocacao[i, j] > 0]
    basicas.append(inicio)  # Inclui a célula 'inicio' como básica provisoriamente

    def dfs(caminho):
        ultimo = caminho[-1]
         # Se o caminho forma um ciclo com pelo menos 4 vértices, retorna-o
        if len(caminho) >= 4 and ultimo == inicio:
            return caminho
        # Percorre as células básicas buscando um ciclo
        for celula in basicas:
            # Evita revisitar a mesma célula (exceto se for o inicio para fechar o loop)
            if celula in caminho and celula != inicio:
                continue
             # Permite movimentação somente na mesma linha ou coluna
            if celula[0] == ultimo[0] or celula[1] == ultimo[1]:
                novo_caminho = caminho + [celula]
                if len(novo_caminho) > 1 and novo_caminho[-1] == inicio and len(novo_caminho) >= 4:
                    return novo_caminho
                resultado = dfs(novo_caminho)
                if resultado is not None:
                    return resultado
        return None

    ciclo = dfs([inicio])
    return ciclo

def ajustar_alocacao(alocacao, ciclo):
    """
    Ajusta a matriz de alocação com base no ciclo encontrado, alternando os sinais
    para redistribuir as quantidades e reduzir o custo total.
    Sinais (+ e -) alternam ao longo do caminho.
    
    Parâmetros:
      - alocacao: Matriz de alocação atual.
      - ciclo: Lista de tuplas que formam o ciclo.
      
    Retorna:
      - nova_alocacao: Matriz de alocação atualizada.
    """
     # Seleciona as posições com sinal negativo (células em posições alternadas, começando na segunda)
    posicoes_menos = ciclo[1::2]
    # Determina o valor máximo que pode ser subtraído (mínimo das alocações nas posições negativas)
    theta = min(alocacao[i, j] for (i, j) in posicoes_menos if alocacao[i, j] > 0)
    nova_alocacao = alocacao.copy()
    sinal = 1
     # Aplica a alternância de sinais (+ e -) ao longo do ciclo
    for (i, j) in ciclo:
        if sinal == 1:
            nova_alocacao[i, j] += theta
        else:
            nova_alocacao[i, j] -= theta
        sinal *= -1
    return nova_alocacao

def transporte_simplex(custo, oferta, demanda, max_iter=100):
    """
    Resolve o problema de transporte utilizando o método Simplex adaptado.
    
    Parâmetros:
      - custo: Matriz de custos.
      - oferta: Vetor de oferta.
      - demanda: Vetor de demanda.
      - max_iter: Número máximo de iterações.
      
    Retorna:
      - alocacao: Matriz de alocação final.
      - custo_total: Custo total da solução.
      - iter_log: Registro das iterações (número da iteração, custo, alocação, tempo decorrido).
    """
    
    tempo_inicial = time.perf_counter_ns()

    total_oferta = oferta.sum()
    total_demanda = demanda.sum()
    matriz_custo = custo.copy()
    vetor_oferta = oferta.copy()
    vetor_demanda = demanda.copy()

    # Balanceamento: se oferta e demanda não são iguais, adiciona linha ou coluna dummy com custo zero
    if total_oferta > total_demanda:
        # Adiciona coluna dummy
        custo_dummy = np.zeros((matriz_custo.shape[0], 1))
        matriz_custo = np.hstack([matriz_custo, custo_dummy])
        vetor_demanda = np.append(vetor_demanda, total_oferta - total_demanda)
    elif total_demanda > total_oferta:
        # Adiciona linha dummy
        custo_dummy = np.zeros((1, matriz_custo.shape[1]))
        matriz_custo = np.vstack([matriz_custo, custo_dummy])
        vetor_oferta = np.append(vetor_oferta, total_demanda - total_oferta)

    # Gera a solução inicial pelo critério do menor custo
    alocacao = solucao_inicial_menor_custo(matriz_custo, vetor_oferta, vetor_demanda)

    custo_inicial = np.sum(alocacao * matriz_custo)
    iter_log = [(0, custo_inicial, alocacao.copy(), 0.0)]

    for iteracao in range(max_iter):
        # Calcula os potenciais u e v para a solução atual
        u, v = calcular_potenciais(matriz_custo, alocacao)
        # Identifica a célula candidata para melhorar a solução
        entrada, delta = encontrar_variavel_entrada(matriz_custo, alocacao, u, v)

        custo_atual = np.sum(alocacao * matriz_custo)
        elapsed = time.perf_counter_ns() - tempo_inicial

        print(f"T: {tempo_inicial} ---- Tm: {time.perf_counter_ns()}")
        # print(f"T: {elapsed} ---- Tm: {elapsed * 1000000000000000000000000000000}")

        elapsed = elapsed / 1000000 #time control # Converte o tempo para milissegundos
         # Registra o estado atual da iteração
        iter_log.append((iteracao + 1, custo_atual, alocacao.copy(), elapsed))
        
        # Se nenhuma variável candidata for encontrada, a solução é ótima
        if entrada is None:
            # Otimalidade alcançada
            break
        # Encontra um ciclo para ajustar a alocação
        ciclo = encontrar_ciclo(alocacao, entrada)
        if ciclo is None:
            # Caso degenerado sem ciclo
            break
        # Ajusta a alocação com base no ciclo encontrado
        alocacao = ajustar_alocacao(alocacao, ciclo)

     # Remove a linha ou coluna dummy, se adicionada, para retornar à estrutura original
    if total_oferta > total_demanda:
        alocacao = alocacao[:, :-1]
        iter_log = [(it, custo, aloc[:, :-1], t) for it, custo, aloc, t in iter_log]
    elif total_demanda > total_oferta:
        alocacao = alocacao[:-1, :]
        iter_log = [(it, custo, aloc[:-1, :], t) for it, custo, aloc, t in iter_log]

    custo_total = np.sum(alocacao * custo)
    return alocacao, custo_total, iter_log
