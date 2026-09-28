# docs/

| Arquivo | Conteúdo |
|---|---|
| `apresentacao_executiva.pdf` | apresentação executiva para a diretoria, com 10 slides (**arquivo de entrega**) |
| `apresentacao_executiva.html` | fonte editável da apresentação (autocontida, com as fontes embutidas); o PDF é gerado a partir dela |
| `roteiro_video.md` | roteiro do vídeo de até 5 minutos, sincronizado com os slides |

Público-alvo: diretoria. Storytelling coeso, sem jargão técnico. Todos os números vêm de
`results/metrics/`.

Para regenerar o PDF depois de editar o HTML (Chrome ou Edge):

```bash
chrome --headless=new --no-pdf-header-footer --print-to-pdf=docs/apresentacao_executiva.pdf docs/apresentacao_executiva.html
```
