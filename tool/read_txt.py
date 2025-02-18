with open("merge_static_share.txt") as f:
    with open("merge_static_share.csv", "w") as f2:
        f2.write("mtx_path,cusparse_static_share,alphasparse_static_share\n")
        for line in f:
            line =line.strip()
            if line.startswith("/home/"):
                mtx_path = line
            if "cusparse" in line:
                cusparse = line.split(": ")[1].split()[0]
            if "alphasparse" in line:
                alphasparse = line.split(": ")[1].split()[0]
            if "require 1e" in line:
                f2.write(f"{mtx_path},{cusparse},{alphasparse}\n")
