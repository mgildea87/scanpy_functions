#Run this in /gpfs/data/fisherlab/conda_envs/scVelo conda env
#This will also have to be run on the cluster via a job or through an interactive session
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from matplotlib import rcParams
import scanpy as sc
import scanpy.external as sce
import anndata
import phate
import scprep
import argparse
import os
import phate



def main(args):

	sc.settings.figdir = ''
	sc.set_figure_params(format = 'png', dpi_save = 500, fontsize = 12, figsize=(10,8))

	cells = load_cells(args.h5ad, args.colors)

	if not args.skip_umap:
		if not os.path.exists('scanpy_umap/'):
			os.makedirs('scanpy_umap')
		cells = umap(cells, args.colors, args.color_id, args.n_pcs, args.n_neighbors)

	if not args.skip_PHATE:
		if not os.path.exists('PHATE/'):
			os.makedirs('PHATE')
		phate(cells, args.colors, args.color_id)

	if not args.skip_PAGA:
		if not os.path.exists('PAGA/'):
			os.makedirs('PAGA')		
		paga(cells, args.colors, args.color_id, args.skip_umap)

	if not args.skip_dpt:
		if not os.path.exists('DPT/'):
			os.makedirs('DPT')	
		dpt(cells, args.color_id, args.root, args.n_dcs, args.colors, args.skip_umap, args.n_pcs, args.n_neighbors, args.n_branch)

def load_cells(h5ad, color_id):

	cells = sc.read_h5ad(h5ad)
	cells.obs['seurat_clusters'] = pd.Categorical(cells.obs['seurat_clusters'])
			
	return cells


def phate(cells, colors, color_id):
	
	if colors != None:
		seurat_cluster_cols = pd.read_csv(colors)
		scc = seurat_cluster_cols['x'].tolist()
	else:
		scc = None

	sce.tl.phate(cells, n_components = 2)
	sce.pl.phate(cells, color = color_id, save = "_%s.png" % (color_id), palette = scc, size = 8)
	os.system('mv phate_%s.png PHATE/phate_%s.png' % (color_id, color_id)) 

	np.savetxt(fname = 'PHATE/PHATE_coor.csv', X = cells.obsm['X_phate'], delimiter = ',')
	np.savetxt("PHATE/PHATE_cell_IDs.csv", cells.obs.index, delimiter=", ", fmt="% s")


def paga(cells, colors, color_id, skip_umap):
	
	if colors != None:
		seurat_cluster_cols = pd.read_csv(colors)
		scc = seurat_cluster_cols['x'].tolist()
	else:
		scc = None
	
	#PAGA
	sc.tl.paga(cells, groups=color_id)
	sc.pl.paga_compare(cells, save = '_umap_%s.png' % (color_id), edge_width_scale=1, solid_edges = 'connectivities_tree', dashed_edges = 'connectivities', palette = scc, size = 8)
	sc.pl.paga_compare(cells, save = '_phate_%s.png' % (color_id), edge_width_scale=1, solid_edges = 'connectivities_tree', dashed_edges = 'connectivities', palette = scc, size = 8, basis = 'phate')
	os.system('mv paga_compare_umap_%s.png PAGA/paga_compare_umap_%s.png' % (color_id, color_id))
	os.system('mv paga_compare_phate_%s.png PAGA/paga_compare_phate_%s.png' % (color_id, color_id))  
	#initialize umap with paga requires umap to be re-run in scanpy. Neighbors are required which for some reason isnt kept during the conversion from seurat to h5ad
	if not skip_umap:
		sc.tl.umap(cells, init_pos = "paga")
		sc.pl.umap(cells, save = '_paga_init_%s.png' % (color_id), color = color_id, palette = scc, size = 8)
		os.system('mv umap_paga_init_%s.png PAGA/umap_paga_init_%s.png' % (color_id, color_id)) 
		sc.pl.paga_compare(cells, save = '_paga_init_%s.png' % (color_id), edge_width_scale=1, solid_edges = 'connectivities_tree', dashed_edges = 'connectivities', palette = scc, size = 8)
		os.system('mv paga_compare_paga_init_%s.png PAGA/paga_compare_paga_init_%s.png' % (color_id, color_id))  

	#export connectivity confidence tables
	pd.DataFrame.sparse.from_spmatrix(cells.uns['paga']['connectivities']).to_csv('PAGA/PAGA_connectivities.csv')
	pd.DataFrame.sparse.from_spmatrix(cells.uns['paga']['connectivities_tree']).to_csv('PAGA/PAGA_connectivities_tree.csv')
	#export paga initialized umap coordinates
	np.savetxt(fname = 'PAGA/PAGA_umap_coor.csv', X = cells.obsm['X_umap'], delimiter = ',')
	np.savetxt("PAGA/PAGA_cell_IDs.csv", cells.obs.index, delimiter=", ", fmt="% s")

def dpt(cells, color_id, root, n_dcs, colors, skip_umap, n_pcs, n_neighbors, n_branch):

	if colors != None:
		seurat_cluster_cols = pd.read_csv(colors)
		scc = seurat_cluster_cols['x'].tolist()
	else:
		scc = None

	#This needs to be experimented with more. e.g. how to use branchings and incorporate with PAGA paths
	#The root cell here is chosen as the first cell in the specified cluster. This should probably be more specific. 
	if skip_umap:
		sc.pp.pca(cells, n_comps = n_pcs)
		sc.pp.neighbors(cells, n_pcs = n_pcs, n_neighbors = n_neighbors, method = 'umap')
	
	sc.tl.diffmap(cells, n_comps = n_dcs)
	cells.uns['iroot'] = np.flatnonzero(cells.obs[color_id] == root)[0]
	sc.tl.dpt(cells, n_dcs = n_dcs, n_branchings = n_branch)
	sc.pl.diffmap(cells, color=[color_id, 'dpt_pseudotime'], save = '_DPT_%s.png' % (color_id), palette = scc, size = 8)
	os.system('mv diffmap_DPT_%s.png DPT/diffmap_DPT_%s.png' % (color_id, color_id)) 

def umap(cells, colors, color_id, n_pcs, n_neighbors):

	if colors != None:
		seurat_cluster_cols = pd.read_csv(colors)
		scc = seurat_cluster_cols['x'].tolist()
	else:
		scc = None
	#Run umap. These parameters were selected to be match seurat's RunUMAP() default parameters
	
	sc.pp.pca(cells, n_comps = n_pcs)
	sc.pp.neighbors(cells, n_pcs = n_pcs, n_neighbors = n_neighbors, method = 'umap')
	sc.tl.umap(cells, min_dist = 0.3)
	sc.pl.umap(cells, save = '_scanpy_%s.png' % (color_id), color = color_id, palette = scc, size = 8)
	os.system('mv umap_scanpy_%s.png scanpy_umap/umap_scanpy_%s.png' % (color_id, color_id)) 

	return cells

def parseArguments():
	
	parser = argparse.ArgumentParser(prog="Run_PHATE", description='')
	required = parser.add_argument_group('required arguments')
	phate = parser.add_argument_group('PHATE options')
	paga = parser.add_argument_group('PAGA options')
	dpt = parser.add_argument_group('Diffustion map and pseudotime options')
	umap = parser.add_argument_group('umap options')

	required.add_argument('-i', '--h5ad', nargs='?', required=True, help='.h5ad file output from seurat. For integrated data sets export the integrated assay from seurat.' , dest='h5ad')
	required.add_argument('-c', '--colors', nargs='?', required=True, help='csv file containing color to cell mappings from seurat. Allows for consistent color palattes between seurat and PHATE, etc..', dest='colors')
	required.add_argument('-c_id', '--color_id', nargs='?', required=True, help='Meta data column in seurat and (now h5ad object) associated with color palette. Most likely "seurat_clusters"', dest='color_id')
	
	phate.add_argument('--skip_PHATE', action='store_true', required=False, help='Skip PHATE', dest='skip_PHATE')

	paga.add_argument('--skip_PAGA', action='store_true', required=False, help='Skip PAGA', dest='skip_PAGA')
	
	umap.add_argument('--skip_umap', action='store_true', required=False, help='Skip umap. If this is skipped the pca->umap will not be recomputed by scanpy. If the object contains umap coordinates from seurat, those will be used by default in other functions', dest='skip_umap')
	umap.add_argument('--n_pcs', nargs='?', default = 30, type = int, required=False, help='Number of principal components to use in umap', dest='n_pcs')
	umap.add_argument('--n_neighbors', nargs='?', default = 30, type = int, required=False, help='Number of neighbors to use in constructing the graph for umap', dest='n_neighbors')

	dpt.add_argument('--skip_dpt', action='store_true', required=False, help='Skip diffusion pseudotime', dest='skip_dpt')
	dpt.add_argument('-r', '--root', nargs='?', default = 0, required=False, help='Root cluster to use for diffusion pseudotime. Has to be consistent with --color_id', dest='root')
	dpt.add_argument('--n_dcs', nargs='?', default = 15, type = int, required=False, help='Number of diffusion components to use in diffusion map and diffusion pseudotime', dest='n_dcs')
	dpt.add_argument('--n_branch', nargs='?', default = 0, type = int, required=False, help='Number of branchings to detect. This parameter is confusing. See scanpy documentation.', dest='n_branch')

	return parser.parse_args()

args = parseArguments()
main(args)