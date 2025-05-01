_fig_iqpe.pdf:
	@for ii in 100 500 1000 2300 ; do \
		python _gen_data.py $${ii} ; \
	done
	@python _plot.py
	
clean:
	$(RM) *.txt _fig_iqpe.pdf