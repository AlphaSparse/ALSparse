import sys
filename = sys.argv[1]

with open(filename) as f:
    with open(filename + ".csv", "w") as f2:
        f2.write("mtx_path,rows,cols,nnz,sparsity,avr_nnz_row,min_nnz_row,max_nnz_row,var_nnz_row,avr_nnz_col,min_nnz_col,max_nnz_col,var_nnz_col,ndiags,diags_ratio,dia_padding_ratio,ell_padding_ratio,empty_rows,empty_cols\n")
        for line in f:
            line =line.strip()
            if line.startswith("mtx:"):
                mtx_path = line.split(": ")[1].split()[0]
            if "rows is" in line:
                rows = line.split(": ")[1].split()[0]
            if "cols is" in line:
                cols = line.split(": ")[1].split()[0]
            if "nnz is" in line:
                nnz = line.split(": ")[1].split()[0]
            if "sparsity is" in line:
                sparsity = line.split(": ")[1].split()[0]
            if "avr_nnz_row is" in line:
                avr_nnz_row = line.split(": ")[1].split()[0]
            if "min_nnz_row is" in line:
                min_nnz_row = line.split(": ")[1].split()[0]
            if "max_nnz_row is" in line:
                max_nnz_row = line.split(": ")[1].split()[0]
            if "var_nnz_row is" in line:
                var_nnz_row = line.split(": ")[1].split()[0]
            if "avr_nnz_col is" in line:
                avr_nnz_col = line.split(": ")[1].split()[0]
            if "min_nnz_col is" in line:
                min_nnz_col = line.split(": ")[1].split()[0]
            if "max_nnz_col is" in line:
                max_nnz_col = line.split(": ")[1].split()[0]
            if "var_nnz_col is" in line:
                var_nnz_col = line.split(": ")[1].split()[0]
            if "diags num is" in line:
                diags_num = line.split(": ")[1].split()[0]
            if "diag_ratio is" in line:
                diag_ratio = line.split(": ")[1].split()[0]
            if "dia_padding_ratio is" in line:
                dia_padding_ratio = line.split(": ")[1].split()[0]
            if "ell_padding_ratio is" in line:
                ell_padding_ratio = line.split(": ")[1].split()[0]
            if "empty rows num is" in line:
                empty_rows_num = line.split(": ")[1].split()[0]
            if "empty cols num is" in line:
                empty_cols_num = line.split(": ")[1].split()[0]
                f2.write(f"{mtx_path},{rows},{cols},{nnz},{sparsity},{avr_nnz_row},{min_nnz_row},{max_nnz_row},{var_nnz_row},{avr_nnz_col},{min_nnz_col},{max_nnz_col},{var_nnz_col},{diags_num},{diag_ratio},{dia_padding_ratio},{ell_padding_ratio},{empty_rows_num},{empty_cols_num}\n")
