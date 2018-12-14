let g:neomake_mymake_maker = {
	\ 'exe': 'make',
	\ 'errorformat': '%f:%l:%c: %m'
	\ }

let g:neomake_verbose=3
let g:neomake_logfile='/tmp/neomake.log'

nnoremap ,r :Neomake! mymake<CR>

let g:ycm_python_binary_path = 'python'

function CythonGoToWrapperHeader()
	let cur_ext = expand('%:e')
	if cur_ext == 'h' || cur_ext == 'cpp' || cur_ext == 'pyx'
		exe 'edit' "wrapper/".expand('%:t:r').".pxd"
	endif
endfunction

function CythonGoToWrapperSource()
	let cur_ext = expand('%:e')
	if cur_ext == 'h' || cur_ext == 'cpp' || cur_ext == 'pxd'
		exe 'edit' "wrapper/".expand('%:t:r').".pyx"
	endif
endfunction

function CppGoToSourceHeader()
	let cur_ext = expand('%:e')
	let ext = "h"
	if cur_ext == ext
		let ext = "cpp"
	endif

	exe 'vsplit' expand('%:r').".".ext
endfunction

nnoremap ,gc :call CppGoToSourceHeader()<CR>
