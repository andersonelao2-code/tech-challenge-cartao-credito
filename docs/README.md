# docs/

| Arquivo | Conteúdo |
|---|---|
| `apresentacao_executiva.pdf` | apresentação executiva para a diretoria, com 10 slides (**arquivo de entrega**) |
| `apresentacao_executiva.html` | fonte editável da apresentação (autocontida, com as fontes embutidas); o PDF é gerado a partir dela |
| `roteiro_video.md` | roteiro do vídeo de até 5 minutos, sincronizado com os slides |
| `apresentacao_video_grupo12.pptx` | slides usados na gravação do vídeo (10 slides, gráficos editáveis, fala de cada slide nas anotações do orador) |
| `apresentacao_video_grupo12.pdf` | os mesmos slides em PDF |
| `roteiro_apresentacao_video.pdf` / `.md` | roteiro da gravação: tempo, integrante, fala e o que mostrar em cada slide |

Público-alvo: diretoria. Storytelling coeso, sem jargão técnico. Todos os números vêm de
`results/metrics/`.

Para regenerar o PDF depois de editar o HTML (Chrome ou Edge):

```bash
chrome --headless=new --no-pdf-header-footer --print-to-pdf=docs/apresentacao_executiva.pdf docs/apresentacao_executiva.html
```
