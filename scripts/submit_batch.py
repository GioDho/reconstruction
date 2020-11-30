#!/bin/env python
# USAGE: python3.8 scripts/submit_batch.py $PWD "[3792-3796]" --outdir cosmics_1stset_261120 --dry-run
# remove --dry-run to submit for real (otherwise only the scripts are created and commands are printed)

jobstring  = '''#!/bin/bash
ulimit -c 0 -S
ulimit -c 0 -H
set -e
cd CYGNOBASE
source scripts/activate_cygno_lngs.sh
RECOSTRING
'''

import os, sys, re

if __name__ == "__main__":
    
    from optparse import OptionParser
    parser = OptionParser(usage='%prog workdir runs [options] ')
    parser.add_option(        '--dry-run',  dest='dryRun',   action='store_true', default=False, help='Do not run the job, only print the command');
    parser.add_option(        '--outdir',   dest='outdir',   type="string", default=None, help='outdirectory');
    parser.add_option(        '--nthreads', dest='nthreads', type="string", default=24, help='number of threads / job');
    parser.add_option('-q',   '--queue',    dest='queue',    type="string", default='cygno-custom', help='queue to be used for the jobs');
    (options, args) = parser.parse_args()

    if len(args)<2:
        parser.print_help()
        exit(1)

    abswpath  = os.path.abspath(args[0]) 
    if not os.path.isdir(abswpath):
        raise RuntimeError('ERROR: {p} is not a valid directory. This is the base dir where the jobs run'.format(p=abswpath))

    nThreads = int(options.nthreads)
    runr = args[1]
    p = re.compile('\[(\d+)-(\d+)\]')
    m = p.match(runr)
    runs = []
    if m:
        minr=m.group(1); maxr=m.group(2)
        if not str.isdigit(minr) or not str.isdigit(maxr):
            raise RuntimeError ("not a good formatted run range: {rr}. Aborting.".format(rr=runr))
        minr=int(minr); maxr=int(maxr)
        if minr>maxr:
            raise RuntimeError ("Warning: first run > end run: {rr}. Aborting.".format(rr=runr))
        runs = list(range(minr,maxr+1))
        print("Will run reconstruction on run range ",runs)
    else:
        raise RuntimeError ("not a good formatted run range string: {rr}. Aborting.".format(rr=runr))

    if not options.outdir:
        raise RuntimeError('ERROR: give at least an output directory. This is where the jobs configs and logs are stored')
    else:
        absopath  = os.path.abspath(options.outdir)
        if not os.path.isdir(absopath):
            print ('making a directory ',absopath,' and running in it')
            os.system('mkdir -p {od}'.format(od=absopath))

    jobdir = absopath+'/jobs/'
    if not os.path.isdir(jobdir):
        os.system('mkdir {od}'.format(od=jobdir))
    logdir = absopath+'/logs/'
    if not os.path.isdir(logdir):
        os.system('mkdir {od}'.format(od=logdir))

    # typically for LIME images (~4MP) MEM = 600MB. 1GB for safety
    if options.queue!='cygno-custom':
        nThreads = min(8,nThreads)
    RAM = nThreads*1000
    ssdcache_opt = 'nodes=1:disk10,' if options.queue=='cygno-custom' else ''
    commands = []
    for run in runs:
        job_file_name = jobdir+'/job_run{r}.sh'.format(r=run)
        log_file_name = logdir+'/job_run{r}.log'.format(r=run)
        tmp_file = open(job_file_name, 'w')

        tmp_filecont = jobstring
        cmd = 'python3.8 reconstruction.py configFile.txt -r {r} -j {nt} '.format(r=run,nt=nThreads)
        tmp_filecont = tmp_filecont.replace('RECOSTRING',cmd)
        tmp_filecont = tmp_filecont.replace('CYGNOBASE',abswpath+'/')
        tmp_file.write(tmp_filecont)
        tmp_file.close()
        
        sub_cmd = 'qsub -q {queue} -l {ssd}ncpus={nt},mem={ram}mb -d {dpath} -e localhost:{logf} -o localhost:{logf} {jobf}'.format(dpath=abswpath,logf=log_file_name,jobf=job_file_name,nt=nThreads,ram=RAM,ssd=ssdcache_opt,queue=options.queue)
        commands.append(sub_cmd)

    if options.dryRun:
        for c in commands:
            print (c)
    else:
        for c in commands:
            os.system(c)

    print ("DONE")


        
