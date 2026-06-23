import re
import os
from json import loads, dump
import MDAnalysis as mda
import requests
from bs4 import BeautifulSoup
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE
import traceback
import argparse as ap
lipids_sel = 'resname DLPE DMPC GPC LPPC PALM PC PGCL POPC POPE DPPC CHL1 CHL'

def json_dict(path):
	"""Converts json file to pyhton dict."""
	json_file=open(path)
	json_str = json_file.read()
	json_data = loads(json_str)
	return json_data

def fix_recname(s):
	"""
	Fix receptor name before saving it. Change sub tags by claudators and put real greek letters instead of HTML entity
	"""
	s = s.replace(' receptor', 'R').replace('<sub>','_{').replace('</sub>','}').replace('adrenoceptor','AR')
	s = re.sub('<.*?>', '', s)

	return(s)

def gpcrdb_info(pdb_set, analrespath, do_our_states=False):
	"""
	Obtain information from GPCRdb for this set of PDBids
	"""

	# If we have information locally avaliable avoid loading it again
	jsonpath = analrespath+"gpcrdb_info_2ndround.json"
	if os.path.exists(jsonpath):
		gpcrdb_dict = json_dict(jsonpath)
		return(gpcrdb_dict)
	
	# Obtain GPCRdb information of our structures
	gpcrdb_data = requests.get('http://gpcrdb.org/services/structure/').json()
	gpcrdb_dict = { entry['pdb_code'] : entry for entry in gpcrdb_data if entry['pdb_code'] in pdb_set }
	
	if do_our_states:
		# Get state of static structures (according to our criteria). INtegrate into GPCRdb dict
		our_states = json_dict(analrespath+'new_activation_inactivation/static_states.json')
		for (pdbcode,ourstate) in our_states.items():
			gpcrdb_dict[pdbcode]['Our_states'] = ourstate

	# Prepare ligand  and GPCR-class entries
	gpcr_classes = {'001':'A','002':'B1','003':'B2','004':'C','005':'D1','006':'F','007':'T','008':'other'}
	for (pdbid, entry) in gpcrdb_dict.items():
		if len(entry['ligands']): # Not all structures have ligands
			gpcrdb_dict[pdbid]['ligand_type'] = entry['ligands'][0]['type']
			gpcrdb_dict[pdbid]['ligand_function'] = entry['ligands'][0]['function']

		gpcrdb_dict[pdbid]['class'] = gpcr_classes[entry['family'].split('_')[0]]

	# Find name of receptor family for each entry (endogenous ligand kind of??)
	for (pdbid, entry) in gpcrdb_dict.items():
		slug = entry['family']
		subfam_slug = '_'.join(slug.split("_")[0:3])

		# Get subfamily from GPCRdb swagger
		response = requests.get("https://gpcrdb.org/services/proteinfamily/"+subfam_slug).json()
		subfam = response['name'].replace('receptors','').replace(' ','_').strip()
		gpcrdb_dict[pdbid]['subfamily'] = subfam[:-1] if subfam[-1] == '_' else subfam # remove last underscore

		# Get GPCR receptor type
		response = requests.get("https://gpcrdb.org/services/protein/"+gpcrdb_dict[pdbid]['protein']).json()
		rectype = gpcrdb_dict[pdbid]['protein'].split('_')[0]
		receptor_name = fix_recname(response['name'])
		# rectype_subbed = fix_rectype(rectype)
		gpcrdb_dict[pdbid]['rectype'] = rectype	 
		gpcrdb_dict[pdbid]['receptor_name'] = receptor_name
		
	# Write contents of dictionary in a file
	with open(jsonpath, "w") as outfile:
		dump(gpcrdb_dict, outfile, indent=4)
		
	return(gpcrdb_dict)

def get_resids_helices(gennum_dict, helices=['2','3','4','5','6','7']):
	"""
	Filter resids belonging to helices (only one character before dot)
	"""
	resids = ' '
	for (resid,gennum) in gennum_dict.items():
		gennum_fix = re.sub('\.\d+','',gennum)
		gennum_split = gennum_fix.split('x')
		if (len(gennum_split[0]) == 1) and (gennum_split[0] in helices):
			resids += " "+resid  
	return(resids)

def find_gennum_standard(pdbcode):
    """
    Use GPCRdb to find ResID of selected generic numbering positions in GPCRdb
    """
    
    # Download GPCRdb structure's website, and extract residue table from it
    structure_data = requests.get('https://gpcrdb.org/structure/refined/'+pdbcode).content
    soup = BeautifulSoup(structure_data, 'html.parser')
    table = soup.find('table', attrs={'id':'rotamers'})
    table_body = table.find('tbody')
    rows = table_body.find_all('tr')

    # ANalyze online table with generic numbering
    gennum_dict = {}
    resid_dict = {}
    for row in rows:
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        gennum = re.sub('.\d+x','x',cols[3]) # We preffer second nomenclature type
        resid = cols[1]
        gennum_dict[resid] = gennum
        resid_dict[gennum] = resid
        
    return(gennum_dict,resid_dict)

def find_gennum_unrefined(pdbcode):
    """
    Find the Ballesteros Wanstein nomenclature for structures not yet refined in GPCRdb
    """
    # Generic numbering from this receptor via GPCRdb
    protname = requests.get('https://gpcrdb.org/services/structure/'+pdbcode).json()['protein']
    generic_nums = requests.get('https://gpcrdb.org/services/residues/extended/'+protname+'/').json()

    # Resid as key, standard nomenclature thingy as value
    gennum_dict = {}
    resid_dict = {}
    for pos in generic_nums:
        gennum = pos['display_generic_number']
        resid = str(pos['sequence_number'])
        if gennum:
            gennum = re.sub('.\d+x','x',gennum) # We preffer second nomenclature type
            gennum_dict[resid] = gennum
            resid_dict[gennum] = resid

    return(gennum_dict,resid_dict)

def find_gennum(pdbcode):
    """
    Use one of our three methods to find generic numbering
    """

    try:
        # Get standard GPCR nomenclature
        (gennum_dict,resid_dict) = find_gennum_standard(pdbcode)
    except Exception as e:
        print("No generic numbering obtained for %s because no unrefined structure avaliable. Using alternative method... (not completely relaible!)"%(pdbcode))
        (gennum_dict,resid_dict) = find_gennum_unrefined(pdbcode)
    return(gennum_dict,resid_dict)
    
def find_membr_zcoor(u):
	"""
	Find top and bottom coordinates of membrane in the z-axis
	"""

	# Select Phoshorous atoms from lipids
	psel = u.select_atoms('name N and %s'%lipids_sel)
	zcoords = [ ps.position[2] for ps in psel]

	# Separate top from bottom Phosphorous
	zmean = np.mean(zcoords)
	topcords = []
	botcords = []
	for zcor in zcoords:
		if zcor > zmean:
			topcords.append(zcor)
		else:
			botcords.append(zcor)

	# Return averages for top and bottom surfaces of membrane
	return(np.mean(botcords),np.mean(topcords))

def mass_centers(u, resids_helices, radius="7", frame=1, radius_maps=10):
	"""
	Find protein residues making the inner core of the system
	"""

	# Find frame in which the trajectory is right now. We'll return it as we found it
	auld_frame=u.trajectory.frame
	# Set universe in desired frame for finding mass centers
	for ts in u.trajectory:
		if ts.frame==frame:
			break

	# Find min and max z-coords for P atoms (we do not want resids bellow or above that point)
	(botcords,topcords) = find_membr_zcoor(u)
	
	# Find intermediate z-coords between bottom and top of receptor
	centers = []
	centers_ary = []
	zheigt = topcords-botcords
	for i in range(1,6):
		partial_top = botcords+(zheigt*(i/5)) 
		partial_bot = partial_top-(zheigt/5)
	
		# Select CA atoms within specified heights, and find their center of mass
		inner_ind = u.select_atoms(
			'(prop z >= %s) and (prop z <= %s) and name CA and protein and resid %s'%(partial_bot, partial_top, resids_helices)
		)
		cm = inner_ind.center_of_mass()
		cm_index = [str(a) for a in cm]
		centers_ary.append(cm_index)
		cm_index = ' '.join(cm_index)
		centers.append(cm_index)
	
	# Make a "around point" selection around each CM obtained (MDanalysis)
	# point_list = ["point "+cm+' '+radius  for cm in centers]
	# point_sel_mda = ' or '.join(point_list)
	
	# Make a "around point" selection around each CM obtained (MDanalysis)
	point_list = [ "(sqr(x-%s)+sqr(y-%s)+sqr(z-%s) < sqr(%s))"%(cm[0],cm[1],cm[2],radius) for cm in centers_ary ]
	lipins_sel_vmd = lipids_sel+' and ('+' or '.join(point_list)+')'
	point_list = [ "(sqr(x-%s)+sqr(y-%s)+sqr(z-%s) < sqr(%s))"%(cm[0],cm[1],cm[2],radius_maps) for cm in centers_ary ]
	# if we wish to select the whole lipid (and not only the lipid's inserted atom) for the VolMaps....
	if whole_res:
		map_sel_vmd = 'same residue as (%s and ('%lipids_sel+' or '.join(point_list)+'))'
	else:
		map_sel_vmd = '%s and ('%lipids_sel+' or '.join(point_list)+')'

	# Rewind universe to original frame
	u.trajectory.rewind()
	for ts in u.trajectory:
		if ts.frame==auld_frame:
			break

	# Return centers of coordinates of different sections of the protein
	return(map_sel_vmd,lipins_sel_vmd)

def calculate_mass_centers(u,section_resids,gpcr_chain='Z'):
	"""
	Calculate and return mass centers of the CA of the submited resids
	"""

	centers = []
	centers_ary = []
	for resids in section_resids:
	
		# Select CA atoms within specified heights, and find their center of mass
		inner_ind = u.select_atoms(
			'protein and name CA and chainID %s and resid %s'%(gpcr_chain, resids)
		)
		cm = inner_ind.center_of_mass()
		cm_index = [str(round(a,2)) for a in cm]
		centers_ary.append(cm_index)
		cm_index = ' '.join(cm_index)
		centers.append(cm_index)

	return(centers,centers_ary)

def make_section_resids(u,botcords,topcords,resids_helices, number_cyls=5):
	"""
	Divide interhelix residues in vertical sections
	"""

	# Find intermediate z-coords between bottom and top of receptor
	section_resids = []
	zheigt = topcords-botcords
	ztops = []
	zbots = []

	# Divide the protein by sections along the Z-axis and find resids in those sections
	for i in range(1,number_cyls+1):
		partial_top = round(botcords+(zheigt*(i/number_cyls)),2)
		partial_bot = round(partial_top-(zheigt/number_cyls),2)
		inner_ind = u.select_atoms(
			'(prop z >= %s) and (prop z <= %s) and protein and name CA and resid %s'%(partial_bot, partial_top, resids_helices)
		)
		# Extract resids from selection and save them
		resids = [ str(res.resid) for res in inner_ind]
		section_resids.append(' '.join(resids))
		ztops.append(partial_top)
		zbots.append(partial_bot)

	# Return top and bottom z-indexes for TM1
	return(section_resids)

def most_distant_byhelix(u,section_resids, centers_ary):
	"""
	From the residue slices submitted in section_resids, select the one in each slice and helix most distant 
	from the corresponding center of mass in centers_ary 
	"""

	farest_resids = []
	for (resids,center) in zip(section_resids,centers_ary):
		
		center_int = np.array([float(x) for x in center]) 
		# Try to separate resids of this section by helix
		resids_ary = resids.split(' ')
		prev_resid = int(resids_ary[0])
		resids_helix = []
		resids_by_helix =  []
		for resid in resids_ary:
			inresid = int(resid)
			if abs(inresid-prev_resid)<4:
				resids_helix.append(resid)
			else: 
				resids_by_helix.append(resids_helix)
				resids_helix = [resid]

			prev_resid = inresid
		resids_by_helix.append(resids_helix)

		# Now, calculate which residue of each Helix is more distant to the center of mass
		farest_resids_section = ''
		for resids_helix in resids_by_helix:
			max_dist = 0
			farest_resid = False
			for resid in resids_helix:
				# Get coordinates of CA of that resid
				coords = u.select_atoms("resid %s and name CA"%resid)[0].position
				# Calculate euclidean distance between points
				dist = np.linalg.norm(center_int-coords)
				# Save new longer distance in helix, if it is so
				if dist>max_dist:
					max_dist = dist
					farest_resid = resid

			farest_resids_section += farest_resid+' '

		farest_resids.append(farest_resids_section.rstrip())

	return(farest_resids)

def helix_mass_centers(u, gennum_dict, helices, radius="7", frame=1, gpcr_chain= 'Z'):
	"""
	Find inner core of selected helices in a system
	"""

	# Find frame in which the trajectory is right now. We'll return it as we found it
	auld_frame=u.trajectory.frame
	# Set universe in desired frame for finding mass centers
	for ts in u.trajectory:
		if ts.frame==frame:
			break

	# Find top and bottom membrane coordinates with OPM
	(ztop, zbot) = find_membr_zcoor(u)

	# Find residues of selected helices
	resids_helices = get_resids_helices(gennum_dict, helices)

	# Divide protein in vertical slices and find ResIDs in those slices
	section_resids = make_section_resids(u, ztop, zbot, resids_helices, number_cyls=5)

	# Find centers of mass of residue slices
	(centers, centers_ary) = calculate_mass_centers(u,section_resids,gpcr_chain)

	# Obtain residues, in each slice and helix, that are the farest away from their corresponding CofM as possible
	farest_resids = most_distant_byhelix(u,section_resids, centers_ary)

	# Find new centers of mass of residue slices (after selecting only one for each helix)
	(centers, centers_ary) =  calculate_mass_centers(u,farest_resids,gpcr_chain)
	
	 # Make a "around point" selection around each CM obtained (MDanalysis)
	point_list = [ "(sqr(x-%s)+sqr(y-%s)+sqr(z-%s) < sqr(%s))"%(cm[0],cm[1],cm[2],radius) for cm in centers_ary ]
	lipins_sel_vmd = '%s and ('%lipids_sel+' or '.join(point_list)+')'
	# if we wish to select the whole lipid (and not only the lipid's inserted atom) for the VolMaps....
	if whole_res:
		map_sel_vmd = 'same residue as (%s and ('%lipids_sel+' or '.join(point_list)+'))'
	else:
		map_sel_vmd = lipins_sel_vmd

	# Rewind universe to original frame
	u.trajectory.rewind()
	for ts in u.trajectory:
		if ts.frame==auld_frame:
			break

	# Return centers of coordinates of diff
	return(map_sel_vmd,lipins_sel_vmd)

def find_receptor_chains(pdbcode):
	"""
	Find chains of this system that belong to a GPCR
	"""

	# Extract chain names and length from PDB
	gpcr_chains = []
	gpcr_names = ['ceptor','rhodopsin','smoothened']
	pdbdict = requests.get('https://data.rcsb.org/graphql?query=\
		{entry(entry_id: "'+pdbcode+'") {\
			polymer_entities {\
			  entity_poly{rcsb_sample_sequence_length, pdbx_strand_id}\
			  rcsb_polymer_entity{pdbx_description}\
			}\
		  }\
		}\
	').json()['data']

	for poly in pdbdict['entry']['polymer_entities']:
		# Determine if this polymer (chain) is a GPCR
		uniname = poly['rcsb_polymer_entity']['pdbx_description'].lower()
		chainIds = poly['entity_poly']['pdbx_strand_id'].split(',')
		if any([(name in uniname) for name in gpcr_names]):
			gpcr_chains += chainIds

	return(gpcr_chains)

def compute_volmaps(pdbpath, trajpath, select_line, map_select_line, resids_helices, gennum_dict, outdata, outmap):
	"""
	Create water maps for the submitted simulation using VMD
	"""
	# if not os.path.exists(outmap):
	print("Computing occupancy maps of %s..."%outmap)
	cmd = ['vmd', pdbpath]+[trajpath]+[' -dispdev', 'text']
	vmd = Popen(cmd, stdin=PIPE, universal_newlines=True)
	vmd.stdin.write("\n".join([
	'set molid top',
	'set seltext "name CA and resid %s"'%resids_helices,
	'set ref [atomselect 0 $seltext frame 0]',
	'set sel [atomselect 0 $seltext]',
	'set all [atomselect 0 all]',
	'set n [molinfo 0 get numframes]',
	'# Function to align',
	# 'for { set i 1 } { $i < $n } { incr i } {',
	# '$sel frame $i',
	# '$all frame $i',
	# '$all move [measure fit $sel $ref]',
	# '}',
	'# Set frame with lipid insertions as active one. Otherwise volmaps doesnt work. Also write resids of selected lipids in file',
	'set outfile1 [open "%s" w]'%outdata,
	'set alllipins {}',
	"for { set i 1 } { $i < $n } { incr i } {",
	'   set mysel [atomselect $molid "%s" frame $i]'%select_line,
	'   if {[$mysel num]} {',
	'	   animate goto $i',
	'	   set resids [$mysel get resid]',
	'	   set resids_u [lsort -unique $resids]',
	'	   set resids_ary [split $resids_u " "]',
	'	   foreach lip_resid $resids_ary {',
	'		   lappend alllipins $lip_resid',
	'		   set lipsel [atomselect $molid "protein and within 3 of (resid $lip_resid and %s)" frame $i]'%lipids_sel,
	'		   set lipresids [$lipsel get resid]',
	'		   set lipresids_u [lsort -unique $lipresids]',
	'		   set j [expr {$i-1}]',
	'		   puts $outfile1 "$j,$lip_resid,$lipresids_u"',
	'};};}',
	'set alllipins_u [lsort -unique $alllipins]',
	'close $outfile1',
	'$ref delete',
	'$all delete',
	'$sel delete',
	'set as [atomselect $molid "%s and resid $alllipins_u"]'%map_select_line,
	'volmap occupancy $as -res 0.5 -allframes -combine avg -checkpoint 0 -o %s'%(outmap),
	'quit']))
	vmd.stdin.close()
	vmd.wait()

	# Parse VMD results and put them on a dictionary for future uses
	frame_inslipids = {}
	for line in open(outdata, 'r'):
		line = line.rstrip()
		(frame,lipid,protids) = line.split(',')
		frame_inslipids.setdefault(frame,{})
		frame_inslipids[frame][lipid] = []
		protids_ary = protids.split(' ')
		for protresid in protids_ary:
			gennum = gennum_dict[protresid] if protresid in gennum_dict else protresid
			frame_inslipids[frame][lipid].append(gennum)	

	return(frame_inslipids) 


def plot_lipid_insertions(sysname, plot_folder, input_path, gennum_dict, plot_by='gennum'):
	"""
	Plot lipid insertions results with matplotlib. Put in Y-axis accumulated frames 
	and in X-axis positions, either:
	resid: residue id
	gennum: generic numbering
	helix: by transmembrane helix
	"""

	# Directories and folders
	data = json_dict(input_path)
	os.makedirs(plot_folder,exist_ok=True)

	# Determine several vairables according to type of plot
	plotwidth = 60
	angle = 90
	xticks_size = 15
	if plot_by=='gennum':
		positions = {pos : 0 for pos in gennum_dict.values() if not pos.startswith('8')}	
	elif plot_by=="resid":
		positions = {int(pos) : 0 for pos in gennum_dict.keys() }	
	elif plot_by=="helix":
		positions = {'TM'+str(a) : 0 for a in range(1,8)} # The 7 Transmembrane helices of GPCRs
		plotwidth = 20
		angle = 0
		xticks_size = 30
	
	resid_dict = { gennum : resid for (resid,gennum) in gennum_dict}

	# Count nombur of times each positions appear to be contacting a lipid
	for (trajid,values) in data.items():
		for (frame, lipids) in values.items():
			for (lipres,poses) in lipids.items():
				for pos in poses:
					if (plot_by=='gennum') and ('x' in pos) and not (pos.startswith('8')):
						positions[pos]+=1
					elif (plot_by=='resid'):
						resid = int(resid_dict[pos]) if pos in resid_dict else int(pos)
						positions.setdefault(resid,0)
						positions[resid] += 1
					elif (plot_by=='helix') and ('x' in pos):
						hel = pos.split('x')[0]
						if (len(hel)==1) and (hel != '8'): # Exclude helix-8 and loop contacts. They cannot do insertions
							namehel = 'TM'+hel
							positions.setdefault(namehel,0)
							positions[namehel] += 1   

	# Sort positions dictionary
	positions = dict(sorted(positions.items()))
	
	# Make plot about number of times each position is nearby an interacting lipid
	fig, ax = plt.subplots(figsize=(plotwidth, 10))
	fig.tight_layout()
	ax.bar(list(positions.keys()),list(positions.values()),width=1,)

	# Set bigger font size
	plt.title(sysname+' frequency of contacts with inserted lipids by '+plot_by, fontdict={'fontsize': 20},pad=40)
	plt.rcParams.update({'font.size': 12})

	# Axis labels
	ax.set_ylabel('Number of frames')
	ax.set_xlabel('Receptor '+plot_by)

	# Rotate labels
	for tick in ax.get_xticklabels():
		tick.set_rotation(angle)

	# Font size
	for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_yticklabels()):
		item.set_fontsize(30)
	for item in ax.get_xticklabels():
		item.set_fontsize(xticks_size)

	# Save figure in a file
	plt.savefig(plot_folder+plot_by+'.png', bbox_inches='tight', dpi=300)

def lipid_stats(sysname, infile, outfile):
	"""
	Count number of frames in which each inserted lipid appears inserted
	Put this info in a json
	"""

	lipid_dict = {'total' : {}}
	for trajid in ['1','2','3']:
		inlips_file = open(infile)
		lipid_dict[trajid] = {}
		for line in inlips_file:
			line=line.rstrip()
			(frame, lipids, protids) = line.split(',')
			for lip in lipids.split():
				lipid_dict[trajid].setdefault(lip,0)
				lipid_dict[trajid][lip] += 1
				lipid_dict['total'].setdefault(lip,0)
				lipid_dict['total'][lip] += 1
	
	# Save data
	with open(outfile,'w') as out:
		dump(lipid_dict, out, indent=4)

def merge_maps(mapfiles, avmap):
	"""
	Merge occupancy maps of systems from the same class
	"""

	try:
		# Open a vmd pipe 
		vmd = Popen(['vmd', ' -dispdev', 'text'], stdin=PIPE, universal_newlines=True)
		# Iterate over occupancy maps of inslipids in trajectories
		molid = 0
		trajcounter = 0
		for mapfile in mapfiles:

			# Skip if no mapfile 
			if not os.path.exists(mapfile):
				continue

			if molid == 0:
				vmd.stdin.write("\n".join([
					'mol new {%s} type {dx} first 0 last -1 step 1 waitfor 1 volsets {0}'%mapfile,
					'set mergedmap 0\n'   
				]))
				molid +=1
			else:
				vmd.stdin.write("\n".join([
					'mol new {%s} type {dx} first 0 last -1 step 1 waitfor 1 volsets {0}'%mapfile,
					'set currentmap %d'%molid,
					'voltool add -union -mol1 $currentmap -mol2 $mergedmap',
					'set mergedmap %d\n' % (molid+1),
					'mol delete $currentmap\n'
				]))

				molid += 2
			trajcounter += 1

		mergedmap = '/tmp/merged.dx'
		vmd.stdin.write('voltool write -mol $mergedmap -o %s\n'%(mergedmap))
		vmd.stdin.close()
		vmd.wait()

		# Divide added volmap by the total number of trajectories
		infile = open(mergedmap,'r')
		outfile = open(avmap,'w')
		for line in infile:
			if line[0].isdigit():
				line = line.rstrip()
				nums = line.split()
				new_nums = [ round(float(num)/trajcounter,5) for num in nums ]
				new_nums = [ str(num) for num in new_nums ]
				new_line = ' '.join(new_nums)+'\n'
				outfile.write(new_line)
			else:
				outfile.write(line)

		# Delete merging file
		os.remove(mergedmap)
	
	except Exception as e:
		print("Merged map could not be computed because of %s"%(e))
		print(traceback.format_exc())


def lipid_insertion_advanced(dynid, dyn_dict, radius_minor, radius_major, files_path, outpath, truelip = False):
	"""
	Main section of this whole script. 
	Find lipids inserted in the receptor during the simulation
	Do it by using Francho's method, of looking for centers of coordinates of vertical sections
	of the protein
	"""

	# Output folders
	precompath = outpath+"Precomputed/"
	out_path = '%s/LipMaps_truelip/'%(precompath) if truelip else '%s/LipMaps/'%(precompath)
	os.makedirs(out_path, exist_ok=True)
	other_outpath = "%s%s_other/"%(out_path,dynid)
	os.makedirs(other_outpath, exist_ok=True)

	# Check if all volmaps for this sim are avaliable. If so, skip dynid
	id_sim = dynid.replace('dyn','')
	files_exist = True
	for trajfile in dyn_dict['traj_f']:
		traj_file_id = os.path.basename(trajfile).split('_')[0] # Id of the trajectory file in DyndbFiles
		volmapath = '%s/%s_occupancy_%s.dx'%(out_path,traj_file_id,id_sim)
		if not os.path.exists(volmapath):
			files_exist = False
	if files_exist and not repeat:
		print('Dyn%s volmaps already exist. Skipping....'%dynid)
		return

	# Find generic numbering
	gennums = dyn_dict['gpcr_pdb']
	# Convert generic numbering info in a Resid->gennum dictionary
	gennum_dict = { pos.split('-')[0] : gennum for (gennum,pos) in gennums.items() }
	if not (gennums):
		print('Dyn%s has no generic numbering. Skipping....'%dynid)
		return

	# Find residues from helices
	resids_helices = get_resids_helices(gennum_dict)

	#Skip if no gpcr
	gpcr_chain = dyn_dict['gpcr_chain']
	if not gpcr_chain:
		print('Dyn%s has no GPCR. Skipping...'%(dynid))
		return

	# Iterate over trajectories
	results= {}
	volmaps = []
	for trajfile in dyn_dict['traj_f']:

		# Input files of simulation
		strucpath = files_path+dyn_dict['struc_f']
		topopath = files_path+dyn_dict['topo_f']
		trajpath = files_path+trajfile
		traj_file_id = os.path.basename(trajfile).split('_')[0] # Id of the trajectory file in DyndbFiles

		# Skip if trajectory file is larger than 2Gb, so not to crash Ori
		if os.path.getsize(trajpath) > 2147483648:
			print('Trajectory %s larger than 2Gb. Skipping to avoid Ori crash...'%trajfile)
			continue

		# Output files 
		volmapath = '%s/%s_occupancy_%s.dx'%(out_path,traj_file_id,id_sim)
		volmaps.append(volmapath)
		raw_vmd_path = '%sraw_vmd_%s.csv'%(other_outpath,traj_file_id)

		if os.path.exists(volmapath):
			continue

		print("analyzing system %s. Trajectory %s..."%(dynid,traj_file_id))

		# Load trajectory and topology into MDA universe, and select protein atoms
		u = mda.Universe(strucpath, trajpath)

		# Find middle frame
		midframe = int(u.trajectory.n_frames/2)
		# Find coordinates of centers-of-mass of verical sections of protein
		# (map_sel_vmd, point_sel_vmd) = mass_centers(u_al,resids_helices,radius,midframe)
		(map_sel_vmd_127, point_sel_vmd_127) = helix_mass_centers(u, gennum_dict, ['1','2','7'], radius_minor, gpcr_chain=gpcr_chain)
		(map_sel_vmd_234567, point_sel_vmd_234567) = helix_mass_centers(u, gennum_dict, ['2','3','4','5','6','7'], radius_major, gpcr_chain=gpcr_chain)
		point_sel_vmd = '('+point_sel_vmd_127+') or ('+point_sel_vmd_234567+')'
		map_sel_vmd = '('+map_sel_vmd_127+') or ('+map_sel_vmd_234567+')'

		# Make volumetric map of inserted lipids
		frame_inslips = compute_volmaps(strucpath, trajpath, point_sel_vmd, map_sel_vmd, resids_helices, gennum_dict, raw_vmd_path, volmapath)
			
		# Append results
		results[traj_file_id] = frame_inslips
	
	# Save insertions organized in dictionary
	dic_ins_path = other_outpath+'insertions.json'
	with open(dic_ins_path,'w') as rawfile:
		dump(results, rawfile, indent=4)

	# Calculate number of frames per lipid ResID
	print('lipd stats...')
	inslip_path = other_outpath+'ins_per_lip.json'
	lipid_stats(dynid, raw_vmd_path, inslip_path)
			

############
## Variables
############

parser = ap.ArgumentParser(description="this calculates interaction frequencies for given simulation")
parser.add_argument(
	'--dynids',
	dest='dynids',
	action='store',
	nargs='+',
	default=False,
	help='Dynamic IDs to compute'
)
parser.add_argument(
	'--whole_residue',
	dest='whole_res',
	action='store_true',
	default=False,
	help='When creating lipid insertion volmaps, take all of the inserted lipid atoms. Instead of just the inserted ones'
)
parser.add_argument(
	'--truelipins',
	dest='truelip',
	action='store_true',
	default=False,
	help='Use a more restrictive cylinder radius to only get those lipids that are truly and doubtlessly inserted in the receptor.'
)
parser.add_argument(
	'--repeat',
	dest='repeat',
	action='store_true',
	default=False,
	help='Re-compute insertion data for trajectories with already avaliable results'
)
parser.add_argument(
	'--threads',
	dest='threads',
	action='store',
	default=3,
	help='Maximum number of trajectories to run at the same time'
)

args = parser.parse_args()
whole_res = args.whole_res 
repeat = args.repeat
threads = int(args.threads)
truelip = args.truelip

# Select radius for selection cylinders, according to specified option
radius_minor = 3 if truelip else 4
radius_major = 7 if truelip else 9

# Set paths and files
files_path = ''
mediaroot = ""

# Load Database information from compl_info.json file
db_dict = json_dict(files_path + "Precomputed/compl_info.json")
dynids = { 'dyn'+a for a in args.dynids} if args.dynids else db_dict.keys()

############
## Main code
############

# Iterate over pdbcodes
if __name__ == "__main__":

	results = {}
	pool = mp.Pool(threads)
	for dynid in dynids:

		try:
	
			# Find lipid insertions
			# sysresults.append(lipid_insertion_advanced(sysname,al_strucpath,mytrajid,radius_minor,radius_major,gennum_dict,resids_helices))
			x = pool.apply_async(lipid_insertion_advanced, args=(dynid,
														db_dict[dynid],
														radius_minor,
														radius_major,
														files_path,
														mediaroot,
														truelip,
														))
			# print(x.get()) # Print errors

		except Exception as e:
			print("System %s could not be run because %s"%(dynid,e))
			print(traceback.format_exc())

	pool.close()
	pool.join() 
