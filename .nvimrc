let g:neomake_make_maker = {
	\ 'exe': 'make',
	\ 'errorformat': '%f:%l:%c: %m'
	\ }

let g:neomake_verbose=3
let g:neomake_logfile='/tmp/neomake.log'

nnoremap ,r :Neomake! make<CR>
