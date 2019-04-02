# T1_ApMaq
## Fazendo alterações no projeto
Alguns comandos básicos do git incluem:
* `git branch`
 * `git branch <nome_branch>` - Cria uma nova branch com o nome `<nome_branch>`
* `git checkout`
 * `git checkout <nome_branch>` - Vai para a branch `<nome_branch>`
* `git status` - Verifica o estado dos seus arquivos. Aqui serão listados os arquivos modificados, adicionados e deletados
* `git fetch` - Apenas pergunta se houve modificações no repositório remoto (checa se alguém modificou o repositório no GitHub) e atualiza informaçes locais, sem modificar seu código.
* `git add`
 * `git add <caminho_do_arquivo>` - Para você escolher quais arquivos serão enviados no seu próximo _commit_.
 Aqui, é muito comum o uso de `git add .`, onde o `.` significa _diretório atual_, ou seja, adiciona todos os arquivos modificados/adicionados/deletados do seu diretório atual para o commit. (pense no `ls .`)
* `git commit` - Cria um novo commit em sua branch local (cria um checkpoint no projeto). Para facilitar o entendimento, imagine como se você tirasse uma foto do estado atual do projeto, e essa foto serve para você poder voltar para esse estado a qualquer momento.
* `git pull`
 * `git pull <nome_remoto> <nome_branch>` - Atualiza sua branch atual com alguma branch do \*GitHub _(não é bem isso, mas por enquanto pense assim)_. Ao mesmo tempo, faz a junção das mudanças. *\*merge\**
* `git push`
 * `git push <nome_remoto> <nome_branch>` - Atualiza o repositório no GitHub com suas modificações. Um exemplo do comando seria:
 `git push origin master` (NÃO FAÇA ISSO SEM AVISAR ALGUÉM ANTES, PLZ. VAMOS TORNAR O MUNDO MELHOR)
