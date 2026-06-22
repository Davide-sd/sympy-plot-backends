(a ready-built PDF is available [here](https://codeberg.org/jipmelon/t/src/branch/main/SympyPlottingBackends_out.pdf))
```bash
uv venv
. .venv/bin/activate
uv pip install -r requirements.txt
uv pip install sympy-plot-backends
make latexpdfja # or whatever your preference
```

If you would like to add a table of contents
```bash
wget 'https://github.com/user-attachments/files/28923619/toc.txt'
pdftocio -v SympyPlottingBackends.pdf < toc
```

Some useful ready-made commands you can tweak if you would like to build a table of contents yourself

(from https://github.com/Krasjet/pdf.tocgen/issues/44)
```bash
perl -i -gpe 's/[ ]{8}/    /g' toc 
```

```bash
sed -i '/Sympy Plotting Backends Documentation\|Chapter 1.  Development and Support\|1.5.  Changelog\|1.4.  Tutorials\|1.3.  SPB Modules Reference\|1.1.  Overview\|"3"\|1.2.  Installation\|"CHAPTER"/d' toc 
```

The original recipe.toml file I used for the very first step may also be handy: https://github.com/user-attachments/files/28923679/recipe.txt (you can rename to `recipe.toml`, I changed to `.txt` for GitHub).
