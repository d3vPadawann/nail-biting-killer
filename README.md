readme_content = """# 🖐️ nail-biting-killer

Um aplicativo em Python desenhado para bloquear hábitos indesejados (como roer as unhas), utilizando Visão Computacional para monitorar e alertar quando a mão se aproxima da boca.

A aplicação roda de forma otimizada em segundo plano e exibe um alerta visual intrusivo (tela cheia) sobrepondo outros aplicativos até que o usuário afaste a mão.

## 🚀 Funcionalidades

- **Monitoramento Otimizado:** Utiliza _MediaPipe Holistic_ e _OpenCV_ com FPS limitado e downscaling de resolução para economizar processamento na CPU.
- **Painel de Controle Flutuante:** Interface nativa e discreta em _Tkinter_ para abrir/fechar a visualização de depuração (Debug View) ou encerrar o aplicativo.
- **Debug View Embutida:** Janela em tempo real com renderização de _bounding boxes_, marcação da boca, ponta dos dedos e barra visual de progresso do _debounce_.
- **Alerta Fullscreen:** Bloqueio de tela vermelho e piscante, utilizando janelas sem bordas e sempre no topo (_always-on-top_).
- **Debounce (Anti-Falso Positivo):** Temporizador configurado para não disparar o alerta instantaneamente com movimentos rápidos e casuais (ex: coçar o nariz ou ajeitar o óculos).

## 🛠️ Pré-requisitos

- **Sistema Operacional:** Linux
- **Linguagem:** Python 3.12+
- **Hardware:** Webcam principal funcional

### Dependências do Sistema (Linux)

O pacote de interface gráfica (`tkinter`) costuma vir separado da instalação base do Python na maioria das distribuições Linux. Instale-o via gerenciador de pacotes da sua distribuição.

Exemplo (base Debian/Ubuntu/Mint):

```bash
sudo apt update
sudo apt install python3-tk
```
