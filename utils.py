import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation

def animar_evolucao_alocacao(registro, nome_cenario):
    """
    Gera e salva uma animação (GIF) que ilustra a evolução da alocação
    ao longo das iterações do algoritmo.
    
    Parâmetros:
      - registro: Lista de registros, onde cada registro é uma tupla
                  contendo (iteração, custo, alocação, tempo decorrido).
      - nome_cenario: Nome do cenário, utilizado para nomear o arquivo de saída.
    """
    # Verifica se o registro está vazio; se sim, exibe mensagem e encerra a função
    if not registro:
        print(f"Log vazio para {nome_cenario}")
        return
    
    # Se houver apenas uma entrada no registro, duplica-a para possibilitar a animação
    if len(registro) == 1:
        entry = registro[0]
        registro = [entry, (entry[0]+1, entry[1], entry[2], entry[3])]

    print(f"{nome_cenario}: log possui {len(registro)} entradas.")

    # Define o caminho para salvar o arquivo de saída e cria o diretório se não existir
    save_path = "outputs"
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f"{nome_cenario}_evolution.gif")

    # Cria a figura e o eixo para a animação
    fig, ax = plt.subplots(figsize=(6, 5))

    def update(frame):
        """
        Função de atualização para cada frame da animação.
        Limpa o eixo e desenha um heatmap da alocação com informações da iteração.
        """
        
        ax.clear() # Limpa o eixo para atualizar com novos dados
        # Extrai os dados da iteração atual
        iteracao, custo, alocacao, _ = registro[frame]
        # Plota o heatmap da matriz de alocação com anotações e formatação definida
        sns.heatmap(alocacao, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax, cbar=False)
        # Define o título do gráfico com a iteração e o custo atual
        ax.set_title(f"Iteração {iteracao}\nCusto: {custo:.2f}")
        return [ax]

    # Cria a animação utilizando a função update para cada frame
    anim = animation.FuncAnimation(
        fig, update, frames=range(len(registro)), interval=800, repeat=False
    )
    try:
        # Salva a animação em formato GIF utilizando o writer 'pillow'
        anim.save(file_path, writer='pillow', fps=1)
        print(f"Animação salva em: {file_path}")
    except Exception as e:
        print(f"Erro ao salvar animação para {nome_cenario}: {e}")

    plt.close() # Fecha a figura para liberar recursos

def plotar_comparacao(custo_simplex, custo_tabu, tempo_simplex, tempo_tabu, 
                     aloc_simp, aloc_tabu, nome_cenario):
    """
    Gera e salva um gráfico comparativo contendo:
      - Comparação dos custos totais obtidos pelos métodos Simplex e Tabu.
      - Comparação dos tempos de execução (em milissegundos).
      - Heatmaps das alocações finais para cada método.
    
    Parâmetros:
      - custo_simplex: Custo final do método Simplex.
      - custo_tabu: Custo final do método Tabu.
      - tempo_simplex: Tempo total (em segundos) de execução do Simplex.
      - tempo_tabu: Tempo total (em segundos) de execução do Tabu.
      - aloc_simp: Matriz de alocação final do método Simplex.
      - aloc_tabu: Matriz de alocação final do método Tabu.
      - nome_cenario: Nome do cenário para identificar o arquivo de saída.
    """
    
    # Os tempos já vêm em microssegundos
    tempo_simplex_us = tempo_simplex
    tempo_tabu_us = tempo_tabu

    # Cria uma figura com 4 subgráficos (2 linhas x 2 colunas)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Primeiro subgráfico: comparação dos custos totais
    diff_cost = abs(custo_simplex - custo_tabu)
    barras_custo = ax1.bar(["Simplex", "Tabu"], [custo_simplex, custo_tabu], 
                                color=["blue", "orange"])
    for barra in barras_custo:
        height = barra.get_height()
        ax1.text(barra.get_x() + barra.get_width()/2., height,
                      f'{height:.2f}',
                      ha='center', va='bottom')
    ax1.set_title(f"Custos Finais\nDif.: {abs(custo_simplex - custo_tabu):.2f}")
    ax1.set_ylabel("Custo")
    
    """diff_cost = abs(custo_simplex - custo_tabu)
    ax1.bar(["Simplex", "Tabu"], [custo_simplex, custo_tabu], color=["blue", "orange"])
    ax1.set_ylabel("Custo Total")
    ax1.set_title(f"Comparação de Custos\nDiferença: {diff_cost:.2f}")"""

    # Ajusta a escala do gráfico de tempo
    max_tempo = max(tempo_simplex_us, tempo_tabu_us)
    
    # Configura limites do eixo Y para melhor visualização
    ax2.set_ylim(0, max_tempo * 1.1)  # Adiciona 10% de margem
    
    # Segundo subgráfico: comparação dos tempos de execução
    diff_time = abs(tempo_simplex_us - tempo_tabu_us)
    barras_tempo = ax2.bar(["Simplex", "Tabu"], [tempo_simplex_us, tempo_tabu_us], 
    color=["blue", "orange"])
    for barra in barras_tempo:
        height = barra.get_height()
        ax2.text(barra.get_x() + barra.get_width()/2., height,
                      f'{height:.2f}',
                      ha='center', va='bottom')
    ax2.set_title(f"Tempo de Execução (µs)\nDiferença: {diff_time:.2f} µs")
    ax2.set_ylabel("Tempo (µs)")
    
    """diff_time = abs(tempo_simplex_us - tempo_tabu_us)
    ax2.bar(["Simplex", "Tabu"], [tempo_simplex_us, tempo_tabu_us], color=["blue", "orange"])
    ax2.set_ylabel("Tempo (µs)")
    ax2.set_title(f"Tempo de Execução (µs)\nDiferença: {diff_time:.2f} µs")
    
    # Adiciona os valores exatos sobre as barras com 2 casas decimais
    for i, v in enumerate([tempo_simplex_us, tempo_tabu_us]):
        ax2.text(i, v + max_tempo * 0.02, f'{v:.2f}', ha='center')"""

    # Terceiro subgráfico: heatmap da alocação final do método Simplex
    sns.heatmap(aloc_simp, annot=True, cmap="YlOrRd", cbar=False, ax=ax3, fmt=".1f")
    ax3.set_title("Alocação - Simplex")

    # Quarto subgráfico: heatmap da alocação final do método Tabu
    sns.heatmap(aloc_tabu, annot=True, cmap="YlOrRd", cbar=False, ax=ax4, fmt=".1f")
    ax4.set_title("Alocação - Tabu")

    # Define um título geral para a figura
    fig.suptitle(f"Resultado Final - {nome_cenario}", fontsize=14)

    # Cria o diretório 'outputs' se não existir e salva a figura comparativa
    save_path = "outputs"
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, f"{nome_cenario}_comparison.png")
    plt.tight_layout() # Ajusta o layout para evitar sobreposição
    plt.savefig(filename, dpi=300)
    print(f"Gráfico comparativo salvo em: {filename}")

    plt.close()  # Fecha a figura para liberar recursos

def animar_iteracoes_cenario(historico_custo_simplex, historico_tempo_simplex,
                            historico_custo_tabu, historico_tempo_tabu,
                            nome_cenario):
    
    """
    Gera e salva uma animação (GIF) que mostra a evolução dos custos e do tempo
    de execução ao longo das iterações para os métodos Simplex e Tabu.
    
    Parâmetros:
      - historico_custo_simplex: Lista com o histórico dos custos do método Simplex.
      - historico_tempo_simplex: Lista com o histórico dos tempos (em segundos) do método Simplex.
      - historico_custo_tabu: Lista com o histórico dos custos do método Tabu.
      - historico_tempo_tabu: Lista com o histórico dos tempos (em segundos) do método Tabu.
      - nome_cenario: Nome do cenário para identificar o arquivo de saída.
    """
    
    # Verifica se os históricos de custo estão vazios; se sim, exibe mensagem e encerra a função
    if not historico_custo_simplex and not historico_custo_tabu:
        print("Histórico de custos vazio.")
        return

    # Garante valores mínimos para o histórico de tempo
    historico_tempo_simplex_ms = [max(t, 0.1) for t in historico_tempo_simplex]
    historico_tempo_tabu_ms = [max(t, 0.1) for t in historico_tempo_tabu]

    # Cria uma figura com 2 subgráficos: um para os custos e outro para os tempos
    plt.figure(figsize=(10, 10))

    # Subgráfico para a evolução dos custos
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_title(f"Evolução dos Custos - {nome_cenario}")
    ax1.set_xlabel("Iteração")
    ax1.set_ylabel("Custo")

    # Subgráfico para a evolução dos tempos de execuçãO
    ax2 = plt.subplot(2, 1, 2)
    ax2.set_title(f"Evolução do Tempo (µs) - {nome_cenario}")
    ax2.set_xlabel("Iteração")
    ax2.set_ylabel("Tempo (µs)")

    # Determina o número máximo de iterações considerando ambos os históricos
    max_iter = max(len(historico_custo_simplex), len(historico_custo_tabu))

    # Inicializa linhas vazias que serão atualizadas durante a animação
    (line_simp_cost,) = ax1.plot([], [], marker='o', color='blue', label='Simplex')
    (line_tabu_cost,) = ax1.plot([], [], marker='o', color='orange', label='Tabu')
    (line_simp_time,) = ax2.plot([], [], marker='o', color='blue', label='Simplex')
    (line_tabu_time,) = ax2.plot([], [], marker='o', color='orange', label='Tabu')

    # Adiciona legendas aos gráficos
    ax1.legend()
    ax2.legend()

    # Ajusta os limites dos eixos para os custos com base em todos os valores coletados
    all_costs = historico_custo_simplex + historico_custo_tabu
    if all_costs:
        ax1.set_xlim(0, max_iter - 1)
        ax1.set_ylim(min(all_costs)*0.95, max(all_costs)*1.05)

    # Ajusta os limites dos eixos para os tempos de execução
    all_times = historico_tempo_simplex_ms + historico_tempo_tabu_ms
    if all_times:
        ax2.set_xlim(0, max_iter - 1)
        ax2.set_ylim(0, max(all_times)*1.05 if max(all_times) > 0 else 1)

    def init():
        """
        Função de inicialização para a animação.
        Zera os dados das linhas para começar do zero.
        """
        line_simp_cost.set_data([], [])
        line_tabu_cost.set_data([], [])
        line_simp_time.set_data([], [])
        line_tabu_time.set_data([], [])
        return line_simp_cost, line_tabu_cost, line_simp_time, line_tabu_time

    def animate(i):
        """
        Atualiza os dados das linhas a cada frame (iterações) da animação.
        """
        if i < len(historico_custo_tabu):
            line_tabu_cost.set_data(range(i+1), historico_custo_tabu[:i+1])
        if i < len(historico_custo_simplex):
            line_simp_cost.set_data(range(i+1), historico_custo_simplex[:i+1])

        if i < len(historico_tempo_tabu_ms):
            line_tabu_time.set_data(range(i+1), historico_tempo_tabu_ms[:i+1])
        if i < len(historico_tempo_simplex_ms):
            line_simp_time.set_data(range(i+1), historico_tempo_simplex_ms[:i+1])

        return line_simp_cost, line_tabu_cost, line_simp_time, line_tabu_time

    # Cria a animação utilizando as funções de inicialização e atualização definidas
    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames=max_iter,
        init_func=init, interval=500, blit=True
    )

    # Define o caminho para salvar o GIF e cria o diretório, se necessário
    save_path = "outputs"
    os.makedirs(save_path, exist_ok=True)
    gif_path = os.path.join(save_path, f"{nome_cenario}_evolution_metrics.gif")

    # Salva a animação em formato GIF com 2 quadros por segundo
    anim.save(gif_path, writer='pillow', fps=2)
    print(f"Métricas salvas em: {gif_path}")

    plt.close() # Fecha a figura para liberar recursos
    
def grafico_top_top(nome_cenario, historico_custo_simplex, historico_custo_tabu, 
                    custo_simplex, custo_tabu, tempo_simplex, tempo_tabu):
    """
    Gera o gráfico top top com layout melhorado e sem sobreposição
    """
    if not historico_custo_simplex and not historico_custo_tabu:
        print("Histórico de custos vazio.")
        return

    # Criar figura com layout grid fixo
    fig = plt.figure(figsize=(15, 10))
    
    # Definir os subplots com posições específicas
    ax_evolucao = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax_custos = plt.subplot2grid((2, 2), (1, 0))
    ax_tempos = plt.subplot2grid((2, 2), (1, 1))

    # Plot da evolução dos custos
    ax_evolucao.plot(historico_custo_tabu, 'orange',marker='o', label='Tabu')
    ax_evolucao.plot(historico_custo_simplex,marker='o', label='Simplex')
    ax_evolucao.set_title("Evolução dos Custos")
    ax_evolucao.set_xlabel("Iteração")
    ax_evolucao.set_ylabel("Custo")
    ax_evolucao.legend()
    ax_evolucao.set_xlim(0, 99)
        # Plot dos custos finais
    barras_custo = ax_custos.bar(["Simplex", "Tabu"], [custo_simplex, custo_tabu], 
                                color=["blue", "orange"])
    # Adiciona os valores em cima das barras de custo
    for barra in barras_custo:
        height = barra.get_height()
        ax_custos.text(barra.get_x() + barra.get_width()/2., height,
                      f'{height:.2f}',
                      ha='center', va='bottom')
    ax_custos.set_title(f"Custos Finais\nDif.: {abs(custo_simplex - custo_tabu):.2f}")
    ax_custos.set_ylabel("Custo")
    
    """ # Plot dos custos finais
    ax_custos.bar(["Simplex", "Tabu"], [custo_simplex, custo_tabu], 
                 color=["blue", "orange"])
    ax_custos.set_title(f"Custos Finais\nDif.: {abs(custo_simplex - custo_tabu):.2f}")
    ax_custos.set_ylabel("Custo")

    # Valor de cada barra
    if(custo_simplex > custo_tabu):
        height = custo_simplex
    else:
        height = custo_tabu
    for i, value in enumerate([custo_simplex, custo_tabu]):
        ax_custos.text(i, value + ((height/100) * 3), str(value), ha='center', va='top', fontsize=12, color='black')"""
    

    # Plot dos tempos
    barras_tempo = ax_tempos.bar(["Simplex", "Tabu"], [tempo_simplex, tempo_tabu], 
                                color=["blue", "orange"])
    # Adiciona os valores em cima das barras de tempo
    for barra in barras_tempo:
        height = barra.get_height()
        ax_tempos.text(barra.get_x() + barra.get_width()/2., height,
                      f'{height:.2f}',
                      ha='center', va='bottom')
    ax_tempos.set_title(f"Tempo de Execução (µs)\nDiferença: {abs(tempo_simplex - tempo_tabu):.2f}")
    ax_tempos.set_ylabel("Tempo (µs)")
    
    # Ajustar layout
    plt.suptitle(f"Análise Comparativa - {nome_cenario}", fontsize=14)
    plt.tight_layout()
    
    """# Plot dos tempos
    ax_tempos.bar(["Simplex", "Tabu"], [tempo_simplex, tempo_tabu], 
                 color=["blue", "orange"])
    ax_tempos.set_title(f"Tempo de Execução (µs)\nDif.: {abs(tempo_simplex - tempo_tabu):.2f}")
    ax_tempos.set_ylabel("Tempo (µs)")

    # Valor de cada barra
    if(tempo_simplex > tempo_tabu):
        height = tempo_simplex
    else:
        height = tempo_tabu
    for i, value in enumerate([tempo_simplex, tempo_tabu]):
        ax_tempos.text(i, value + ((height/100) * 3), str(value), ha='center', va='top', fontsize=12, color='black')"""

    # Salvar
    save_path = "outputs"
    os.makedirs(save_path, exist_ok=True)
    image_path = os.path.join(save_path, f"{nome_cenario}_grafico_top_top.png")
    plt.savefig(image_path)
    print(f"Gráfico top top salvo em: {image_path}")
    plt.close()

