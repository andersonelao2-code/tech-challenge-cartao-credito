# Roteiro do vídeo executivo (até 5 minutos)

Acompanha `apresentacao_executiva.pdf`. A duração é uma estimativa para fala em ritmo natural
(cerca de 130 palavras por minuto). Linguagem executiva, sem jargão técnico. Pelo menos um
integrante precisa aparecer ou narrar.

| Slide | Tempo | Fala |
|---|---|---|
| 1 · Capa | 0:00–0:20 | "Olá, nós somos o Grupo 12 e eu sou [nome]. Neste vídeo vamos mostrar como usamos dados para ajudar uma instituição financeira a decidir onde concentrar a análise de pedidos de cartão de crédito." |
| 2 · O problema | 0:20–0:55 | "Todo cartão aprovado para quem não paga vira prejuízo. E analisar cada pedido com o mesmo cuidado é caro e lento. Na nossa base, só 1,7% dos clientes deixaram de pagar duas faturas seguidas: é uma agulha no palheiro. Por isso a taxa de acerto engana. Quem dissesse que todo mundo é bom pagador acertaria 98% das vezes, sem encontrar um único caso de risco." |
| 3 · Os dados | 0:55–1:25 | "Usamos duas fontes: o cadastro de 438 mil solicitantes e mais de um milhão de registros mensais de pagamento. Como a base não diz quem foi aprovado, definimos como mau pagador quem atrasou 60 dias ou mais, ou seja, deixou de pagar duas faturas seguidas. Atrasos curtos são comuns e não indicam risco real." |
| 4 · Sem perfil óbvio | 1:25–1:50 | "O primeiro achado: não existe um perfil óbvio de mau pagador. Entre os grupos, a taxa varia só entre 1,2% e 2,9%. Nenhuma característica isolada resolve o problema. O sinal está na combinação de várias delas, e é isso que um modelo faz bem." |
| 5 · Cuidados | 1:50–2:30 | "Antes de modelar, tratamos quatro armadilhas: aposentados registrados com um código que parecia mil anos de emprego, ocupação em branco em 31% dos casos, a mesma pessoa cadastrada várias vezes e uma informação que era a própria resposta disfarçada. As 36 mil contas são de menos de 10 mil pessoas, então separamos treino e teste por pessoa, para testar o modelo em quem ele nunca viu. E a resposta disfarçada dava um falso acerto de 100% e foi removida." |
| 6 · A solução | 2:30–3:05 | "Comparamos quatro abordagens nas mesmas condições. A vencedora foi o XGBoost, com nota 0,80 numa escala em que 0,5 é sorteio e 1 seria perfeito. A nota foi medida em pessoas que o modelo nunca tinha visto, e ficou estável em relação ao treino. A Regressão Logística ficou praticamente empatada e é uma alternativa mais simples de explicar." |
| 7 · Resultado prático | 3:05–3:40 | "Este é o resultado mais importante. Se a equipe revisar só os 20% de pedidos que o modelo aponta como mais arriscados, encontra 64% de todos os maus pagadores. Um sorteio encontraria 20%. Com apenas 10% dos pedidos, já encontra 40%, quatro vezes mais que o acaso. É menos esforço, concentrado onde está o risco." |
| 8 · O que pesa | 3:40–4:00 | "O que mais pesa na previsão é o histórico com o banco. Meses quitando o saldo e uso moderado do crédito indicam menor risco. No cadastro, tempo de emprego e renda maiores também indicam menor risco, mas pesam pouco." |
| 9 · Limites | 4:00–4:40 | "O modelo tem limites claros. Se recusasse pedidos sozinho, negaria cerca de 17 bons pagadores para cada mau pagador evitado, por isso a decisão final deve ser humana. Só com os dados do cadastro, a nota cai para 0,59, perto do sorteio: o modelo serve para quem já é cliente, e para quem chega pela primeira vez é preciso usar birôs de crédito. E, como o gênero influencia a previsão, é necessária uma auditoria de equidade antes de qualquer uso real." |
| 10 · Recomendação | 4:40–5:00 | "Nossa recomendação é um piloto: ordenar a fila de análise pelo risco e medir o ganho, definindo o ponto de corte junto com a área de risco. Menos tempo gasto nos pedidos seguros, mais atenção onde o risco está. Obrigado." |

## Dicas de gravação

- Grave em uma única tomada por slide, com a tela da apresentação e a câmera ou a voz.
- Confira a duração final: o limite é de **5 minutos**.
- Publique no YouTube como **não listado**, ou no Google Drive com acesso "qualquer pessoa com o
  link". Teste o link em uma janela anônima.
