"""No-model regression checks for the isolated advanced configuration boundary."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import advanced_guard
from advanced_guard import (ROOT, STOCK, shared, source_hashes, MTP_FIXTURES, MTP_MODES,
                            PARITY_SCOPE, parse_remote_layer_ids, remote_placement,
                            native_expert_payload)

SELECTED_IDS=[22,28,30,34]
SELECTED_PAYLOAD=4535091200


def placement_proof(ids=SELECTED_IDS,payload=SELECTED_PAYLOAD,explicit=True):
    return {'remote_expert_layer_ids_for_dual':list(ids),
            'remote_expert_payload_bytes_for_dual':payload,
            'remote_expert_layer_ids_explicit_for_dual':explicit}


class Boundaries(unittest.TestCase):
    def test_frozen_remote_cache_requires_bound_hybrid_opt_in(self):
        flag='--remote-expert-freeze-cache';hybrid='--remote-expert-hybrid'
        self.assertFalse(advanced_guard.frozen_remote_cache_placement([],False))
        self.assertTrue(advanced_guard.frozen_remote_cache_placement([hybrid,flag],True))
        for argv,expected in (([],True),([hybrid,flag],False),([flag],True),
                              ([hybrid,flag,flag],True),([hybrid,hybrid,flag],True),([],0)):
            with self.subTest(argv=argv,expected=expected),self.assertRaises(ValueError):
                advanced_guard.frozen_remote_cache_placement(argv,expected)

    def test_hybrid_cache_transfer_counters_fail_closed(self):
        hit='strata hybrid experts request: delta 4 layers, GPU0 12 hit entries, GPU1 28 miss entries, 2 cold cache reads'
        stat='strata hybrid cache request: frozen 1; adaptive_reads 0 bytes 0 wall_ms 0.000; restore_reads 2 bytes 4000000 wall_ms 1.250'
        valid=hit+'\n'+stat
        rows=advanced_guard.validate_hybrid_cache_log(valid,1,True)
        self.assertEqual(rows[0]['adaptive_reads'],0)
        self.assertEqual(rows[0]['restore_reads'],2)
        for text,n,frozen in ((valid,2,True),(valid,1,False),(valid+'\n'+stat,1,True),
                              (valid.replace('adaptive_reads 0 bytes 0','adaptive_reads 1 bytes 1'),1,True),
                              (valid.replace('wall_ms 0.000','wall_ms 0.001'),1,True),
                              (valid.replace('2 cold cache','3 cold cache'),1,True),
                              (valid.replace('4000000','99999999999'),1,True),
                              (valid.replace('wall_ms 1.250','wall_ms inf'),1,True),
                              (valid.replace('2 bytes 4000000','0 bytes 0').replace('wall_ms 1.250','wall_ms 0.000')
                               .replace('2 cold cache','0 cold cache'),1,True)):
            with self.subTest(text=text,n=n,frozen=frozen),self.assertRaises((ValueError,RuntimeError)):
                advanced_guard.validate_hybrid_cache_log(text,n,frozen)
        adaptive=valid.replace('frozen 1','frozen 0').replace('adaptive_reads 0 bytes 0 wall_ms 0.000',
                     'adaptive_reads 3 bytes 5000000 wall_ms 2.000').replace('2 cold cache','5 cold cache')
        self.assertEqual(advanced_guard.validate_hybrid_cache_log(adaptive,1,False)[0]['adaptive_reads'],3)

    def test_hybrid_opt_in_and_exact_fixture_markers(self):
        flag='--remote-expert-hybrid'
        self.assertTrue(advanced_guard.hybrid_placement([flag],True))
        self.assertFalse(advanced_guard.hybrid_placement([],False))
        for argv,expected in (([],True),([flag],False),([flag,flag],True),([],0),([flag],'true')):
            with self.subTest(argv=argv,expected=expected),self.assertRaises(ValueError):
                advanced_guard.hybrid_placement(argv,expected)
        valid=advanced_guard.MASKED_MARKER+'\n'+advanced_guard.HYBRID_MARKER+'\n'
        advanced_guard.validate_hybrid_log(valid)
        for log in ('',advanced_guard.MASKED_MARKER,advanced_guard.HYBRID_MARKER,
                    valid+advanced_guard.HYBRID_MARKER,valid.replace('outputs=exact','outputs=tolerance')):
            with self.subTest(log=log),self.assertRaises(RuntimeError):advanced_guard.validate_hybrid_log(log)

    def test_hybrid_requires_bound_sparse_dispatch_and_restore_admission(self):
        gates={**placement_proof(),'passed':True,'executable_sha256':'testexe','source_hashes':{},
               'bound_files':[],'evidence_scope':'unit test','remote_hybrid_for_dual':True}
        parity={'passed':True,'engine_executable_sha256':'testexe','gates_sha256':'testgates',
                'layers':4,'decode_calls':6,'prefill_calls':5,'quant_bytes':'exact','outputs':'exact','blob_bytes':'exact',
                'layer_ids':SELECTED_IDS,'payload_bytes':SELECTED_PAYLOAD,'scope':advanced_guard.REMOTE_SCOPE,
                'masked_decode':{'devices':2,'calls_per_device':4,'entries_per_device':32,
                                 'all_hit_calls_per_device':1,'outputs':'exact','negatives':'exact'},
                'hybrid_dispatch':{'devices':2,'outputs':'exact','cold_restore':'exact','failures':'exact'},
                'bound_evidence':[{'path':str(ROOT/'parity-runs/mock04'/name),'sha256':'testgates'}
                                  for name in ('result.json','native.log','probe.json','command.json')]}
        native_log=advanced_guard.MASKED_MARKER+'\n'+advanced_guard.HYBRID_MARKER
        def read(path,*args,**kwargs):
            if path.name=='native.log':return native_log
            return json.dumps(parity if path.name=='remote-experts-admission.json' else gates)
        with patch.object(shared,'collect_report',return_value={}),patch.object(shared,'EXE_HASH','testexe'), \
             patch.object(shared,'digest',return_value='testgates'),patch.object(advanced_guard,'source_hashes',return_value={}), \
             patch.object(Path,'read_text',read):
            def check():
                return advanced_guard.artifact_checks('experimental-no-vram-floor',require_mtp_equivalence=False)
            report,_=check()
            self.assertTrue(report['hybrid_expert_equivalence_checked'])
            for section,key,bad in (('masked_decode','devices',1),('masked_decode','calls_per_device',3),
                                    ('masked_decode','entries_per_device',0),('masked_decode','outputs','tolerance'),
                                    ('hybrid_dispatch','cold_restore',None),('hybrid_dispatch','failures',None)):
                old=parity[section][key];parity[section][key]=bad
                with self.subTest(section=section,key=key),self.assertRaises(RuntimeError):check()
                parity[section][key]=old
            for bad in ('true',1):
                gates['remote_hybrid_for_dual']=bad
                with self.assertRaises(RuntimeError):check()
            gates['remote_hybrid_for_dual']=True
            for bad in ('true',1):
                gates['remote_cache_frozen_for_dual']=bad
                with self.assertRaises(RuntimeError):check()
            gates['remote_cache_frozen_for_dual']=True
            self.assertTrue(check()[0]['remote_cache_frozen_for_dual'])
            gates['remote_hybrid_for_dual']=False
            with self.assertRaises(RuntimeError):check()
            gates['remote_hybrid_for_dual']=True
            gates['remote_decode_mode_for_dual']='graphs'
            with self.assertRaises(RuntimeError):check()

    def test_profiled_full_trial_requires_exact_bounded_wait_gate(self):
        gates={**placement_proof(),'passed':True,'executable_sha256':'testexe','source_hashes':{},'bound_files':[],
               'evidence_scope':'unit test','matched_verify_wait_profile':True,
               'verify_wait':{'devices':2,'profiled_replays_per_device':4,'ready_per_device':2,
                              'delayed_per_device':2,'timer_unit':'globaltimer_ns','outputs':'exact'}}
        with patch.object(shared,'collect_report',return_value={}),patch.object(shared,'EXE_HASH','testexe'), \
             patch.object(advanced_guard,'source_hashes',return_value={}), \
             patch.object(Path,'read_text',return_value=json.dumps(gates)) as read:
            def check():
                read.return_value=json.dumps(gates)
                return advanced_guard.artifact_checks('experimental-no-vram-floor',
                         require_mtp_equivalence=False,require_remote_experts=False)
            report,_=check()
            self.assertTrue(report['verifier_wait_profiling'])
            for key,value in (('devices',1),('profiled_replays_per_device',3),('ready_per_device',1),
                              ('delayed_per_device',1),('timer_unit','cycles'),('outputs','tolerance')):
                previous=gates['verify_wait'][key]
                gates['verify_wait'][key]=value
                with self.assertRaises(RuntimeError):check()
                gates['verify_wait'][key]=previous
            gates['matched_verify_wait_profile']='true'
            with self.assertRaises(RuntimeError):check()

    def test_expert_profile_requires_both_numerical_admissions(self):
        from run_mtp_equivalence import test_spec
        prefix,marker,mtp,remote,scope=test_spec('expert-profile')
        self.assertEqual(prefix,'remote_experts_profile')
        self.assertEqual(marker,advanced_guard.DECODE_MARKER)
        self.assertTrue(mtp and remote)
        self.assertIn('no main host expert arena',scope)
        self.assertFalse(test_spec('equivalence')[2] or test_spec('equivalence')[3])
        self.assertTrue(test_spec('remote-experts')[2])
        self.assertFalse(test_spec('remote-experts')[3])
        with self.assertRaises(ValueError):
            test_spec('other')

    def test_decode_profile_requires_distinct_exact_counted_conditions(self):
        from run_mtp_equivalence import validate_decode_profile
        rows=[{'device':device,'mode':mode,'profiling':profiling,'gap_ms':gap,'bursts':64,'calls':256,
               'graph_replays':256 if mode=='graphs' else 0,'profile_calls':256 if profiling else 0,
               'reference_bytes':65536000,'input_bytes':40960,'output_bytes':409600,'digest':'same',
               'layer_ids':'22,28,30,34','payload_bytes':SELECTED_PAYLOAD,
               'active_decode_wall_ms':1.0,'timing_scope':'decode_calls_only_checks_and_pauses_excluded'}
              for device in (0,1) for mode in ('original','packed','graphs')
              for profiling in (False,True) for gap in (0,40)]
        marker='PASS remote_experts_profile bursts=64 layers=4 modes=3 conditions=24 ids=22,28,30,34 outputs=exact'
        def log():
            return '\n'.join(json.dumps(row) for row in rows)+'\n'+marker
        self.assertEqual(len(validate_decode_profile(log(),SELECTED_IDS,SELECTED_PAYLOAD,marker)),24)
        for key,value in (('graph_replays',1),('reference_bytes',100),('digest','other'),
                          ('active_decode_wall_ms',float('nan')),('calls',255),('mode','other'),
                          ('layer_ids',SELECTED_IDS),('payload_bytes',3250585600)):
            previous=rows[0][key]
            rows[0][key]=value
            with self.assertRaises(RuntimeError):
                validate_decode_profile(log(),SELECTED_IDS,SELECTED_PAYLOAD,marker)
            rows[0][key]=previous
        previous=rows[-1]
        rows[-1]=rows[0]
        with self.assertRaises(RuntimeError):
            validate_decode_profile(log(),SELECTED_IDS,SELECTED_PAYLOAD,marker)
        rows[-1]=previous

    def test_ordinary_stock_policy_unchanged(self):
        p=shared.policy('default')
        self.assertEqual((p['ram_floor_gib'],p['commit_floor_gib'],p['job_cap_gib']),(12,16,48))
        self.assertEqual((p['gpu_start_floor_mib'],p['gpu_floor_mib']),(14336,2048))

    def test_advanced_finite_exception_keeps_emergency_checks(self):
        with patch.object(shared,'ROOT',ROOT):
            p=shared.policy('experimental-no-vram-floor',3072)
        self.assertEqual((p['ram_floor_gib'],p['commit_floor_gib'],p['job_cap_gib']),(0,4,60))
        self.assertEqual((p['gpu_start_floor_mib'],p['gpu_floor_mib']),(0,0))
        shared.check_resources({'ram_free':0,'commit_free':4*shared.GiB},[[0,0,79],[1,0,79]],p)
        with self.assertRaises(RuntimeError):
            shared.check_resources({'ram_free':0,'commit_free':4*shared.GiB-1},[[0,0,79],[1,0,79]],p)
        with self.assertRaises(RuntimeError):
            shared.check_resources({'ram_free':0,'commit_free':4*shared.GiB},[[0,0,80],[1,0,79]],p)

    def test_visibility_is_bounded_before_any_launch(self):
        with self.assertRaises(ValueError):
            shared.run_trial(SimpleNamespace(gpu=0),device_visibility='0,1,2')

    def test_small_parity_job_does_not_widen_stock_cap_whitelist(self):
        with self.assertRaises(ValueError):
            shared.make_job(8)

    def test_selected_layer_parser_payload_and_ambiguity(self):
        self.assertEqual(parse_remote_layer_ids('22,28,30,34'),SELECTED_IDS)
        self.assertEqual(parse_remote_layer_ids('22,24,28,30,33,34'),advanced_guard.SIX_REMOTE_LAYERS)
        for value in ('','22,28,30','22,28,30,34,35','22,28,28,34','28,22,30,34',
                      '-1,22,30,34','22,30,34,48','22, 28,30,34','a,28,30,34',
                      '٢٢,28,30,34','22,24,28,30,33,34,35','22,24,28,30,33,34,35,36',
                      '22,24,28,30,33,33','22,24,28,30,34,33','22,24,28,30,33,48'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                parse_remote_layer_ids(value)
        self.assertEqual(remote_placement(['--remote-expert-layer-ids','22,28,30,34']),
                         (SELECTED_IDS,True))
        self.assertEqual(remote_placement(['--remote-expert-layer-ids','22,24,28,30,33,34']),
                         (advanced_guard.SIX_REMOTE_LAYERS,True))
        self.assertEqual(remote_placement(['--remote-expert-layers','4']),([0,1,2,3],False))
        self.assertEqual(remote_placement(['--remote-expert-layers','0']),([],False))
        for argv in ([],['--remote-expert-layers','4','--remote-expert-layer-ids','22,28,30,34'],
                     ['--remote-expert-layer-ids','22,28,30,34','--remote-expert-layer-ids','22,28,30,34'],
                     ['--remote-expert-layers','3'],['--remote-expert-layers','6']):
            with self.subTest(argv=argv),self.assertRaises(ValueError):remote_placement(argv)
        base=json.loads((STOCK/'stock-config.json').read_text())
        pack=Path(base['args'][base['args'].index('--pack')+1])
        self.assertEqual(native_expert_payload(pack,[0,1,2,3]),3250585600)
        self.assertEqual(native_expert_payload(pack,SELECTED_IDS),SELECTED_PAYLOAD)
        self.assertEqual(native_expert_payload(pack,advanced_guard.SIX_REMOTE_LAYERS),
                         advanced_guard.SIX_REMOTE_PAYLOAD)

    def test_six_marker_is_exact_and_original_only(self):
        marker=advanced_guard.REMOTE_SIX_MARKER
        advanced_guard.validate_six_expert_log(marker+'\n')
        for log in ('',marker+'\n'+marker,marker.replace('layer_coverage=all','layer_coverage=partial'),
                    marker.replace('outputs=exact','outputs=tolerance'),
                    marker.replace('allocation_bytes=6949138728','allocation_bytes=6949138727'),
                    marker+'\n'+advanced_guard.MASKED_MARKER,marker+'\n'+advanced_guard.HYBRID_MARKER):
            with self.subTest(log=log),self.assertRaises(RuntimeError):
                advanced_guard.validate_six_expert_log(log)

    def test_six_diagnostic_rejects_four_layer_optimized_profile(self):
        from run_mtp_equivalence import diagnostic_placement
        ids=advanced_guard.SIX_REMOTE_LAYERS
        gates=placement_proof(ids,advanced_guard.SIX_REMOTE_PAYLOAD)
        config={'strategy':'resident-experts','args':['--remote-expert-layer-ids',','.join(map(str,ids))]}
        self.assertEqual(diagnostic_placement(config,gates,'remote-experts'),
                         (ids,advanced_guard.SIX_REMOTE_PAYLOAD,True))
        with self.assertRaises(ValueError):diagnostic_placement(config,gates,'expert-profile')

    def test_six_admission_requires_bound_actual_host_log_and_all_layer_coverage(self):
        host_path=ROOT/'gate-runs/mock06/compacted-loader.log'
        gates={**placement_proof(advanced_guard.SIX_REMOTE_LAYERS,advanced_guard.SIX_REMOTE_PAYLOAD),
               'passed':True,'executable_sha256':'testexe','source_hashes':{},'evidence_scope':'unit test',
               'bound_files':[{'path':str(host_path),'sha256':'testgates'}],
               'compacted_loader':{'selected6_host_layout':dict(advanced_guard.SIX_HOST_PROOF),
                                   'log':str(host_path)}}
        parity={'passed':True,'engine_executable_sha256':'testexe','gates_sha256':'testgates',
                'layers':6,'decode_calls':6,'prefill_calls':7,'quant_bytes':'exact','outputs':'exact','blob_bytes':'exact',
                'layer_ids':list(advanced_guard.SIX_REMOTE_LAYERS),'payload_bytes':advanced_guard.SIX_REMOTE_PAYLOAD,
                'scope':advanced_guard.REMOTE_SIX_SCOPE,
                'six_layer_coverage':dict(advanced_guard.SIX_EXPERT_PROOF),
                'bound_evidence':[{'path':str(ROOT/'parity-runs/mock06'/name),'sha256':'testgates'}
                                  for name in ('result.json','native.log','probe.json','command.json')]}
        logs={'compacted-loader.log':advanced_guard.SIX_HOST_MARKER,'native.log':advanced_guard.REMOTE_SIX_MARKER}
        def read(path,*args,**kwargs):
            if path.name in logs:return logs[path.name]
            return json.dumps(parity if path.name=='remote-experts-admission.json' else gates)
        with patch.object(shared,'collect_report',return_value={}),patch.object(shared,'EXE_HASH','testexe'), \
             patch.object(shared,'digest',return_value='testgates'),patch.object(advanced_guard,'source_hashes',return_value={}), \
             patch.object(Path,'read_text',read):
            def check():
                return advanced_guard.artifact_checks('experimental-no-vram-floor',require_mtp_equivalence=False)
            report,_=check()
            self.assertTrue(report['remote_expert_equivalence_checked'])
            self.assertTrue(report['six_layer_expert_equivalence_checked'])
            for section,key,bad in ((gates,'remote_expert_layer_ids_for_dual',[21,24,28,30,33,34]),
                                    (gates,'remote_expert_payload_bytes_for_dual',advanced_guard.SIX_REMOTE_PAYLOAD-1),
                                    (gates,'remote_expert_layer_ids_explicit_for_dual',False),
                                    (gates,'remote_decode_mode_for_dual','packed'),
                                    (gates,'remote_decode_mode_for_dual','graphs'),
                                    (gates,'remote_hybrid_for_dual',True),(gates,'remote_cache_frozen_for_dual',True),
                                    (gates,'bound_files',[]),
                                    (parity,'layers',4),(parity,'prefill_calls',5),(parity,'scope',advanced_guard.REMOTE_SCOPE),
                                    (parity,'six_layer_coverage',{})):
                old=section.get(key);had=key in section;section[key]=bad
                with self.subTest(key=key,bad=bad),self.assertRaises(RuntimeError):check()
                if had:section[key]=old
                else:del section[key]
            for proof,key,bad in ((gates['compacted_loader']['selected6_host_layout'],'devices',1),
                                  (gates['compacted_loader']['selected6_host_layout'],'holes',5),
                                  (gates['compacted_loader']['selected6_host_layout'],'aliases','tolerance'),
                                  (parity['six_layer_coverage'],'allocation_bytes',advanced_guard.SIX_REMOTE_ALLOCATION-1),
                                  (parity['six_layer_coverage'],'blob_cases_per_device',11),
                                  (parity['six_layer_coverage'],'all_layer_coverage','partial'),
                                  (parity['six_layer_coverage'],'mode','graphs'),
                                  (parity['six_layer_coverage'],'max_prefill_t',1024)):
                old=proof[key];proof[key]=bad
                with self.subTest(key=key,bad=bad),self.assertRaises(RuntimeError):check()
                proof[key]=old
            old=gates['compacted_loader']['log']
            for bad in (str(STOCK/'gate-runs/mock06/compacted-loader.log'),str(host_path.with_name('other.log'))):
                gates['compacted_loader']['log']=bad
                with self.assertRaises(RuntimeError):check()
            gates['compacted_loader']['log']=old
            for name in logs:
                good=logs[name]
                for bad in ('',good+'\n'+good,good.replace('exact','tolerance')):
                    logs[name]=bad
                    with self.subTest(name=name,bad=bad),self.assertRaises(RuntimeError):check()
                logs[name]=good

    def test_mtp_runtime_uses_prefix_numerical_fixture_without_remote_placement(self):
        from run_mtp_equivalence import diagnostic_placement
        config={'strategy':'mtp','args':['--remote-expert-layers','0']}
        gates={**placement_proof([0,1,2,3],3250585600,False)}
        self.assertEqual(diagnostic_placement(config,gates,'remote-experts'),
                         ([0,1,2,3],3250585600,False))
        self.assertEqual(diagnostic_placement(config,gates,'expert-profile'),
                         ([0,1,2,3],3250585600,False))
        self.assertEqual(diagnostic_placement(config,gates,'equivalence'),([],0,False))

    def test_bound_placement_proof_is_strict(self):
        gates={**placement_proof(),'passed':True,'executable_sha256':'testexe','source_hashes':{},
               'bound_files':[],'evidence_scope':'unit test'}
        with patch.object(shared,'collect_report',return_value={}),patch.object(shared,'EXE_HASH','testexe'), \
             patch.object(advanced_guard,'source_hashes',return_value={}), \
             patch.object(Path,'read_text',side_effect=lambda *a,**k:json.dumps(gates)):
            def check():
                return advanced_guard.artifact_checks('experimental-no-vram-floor',
                         require_mtp_equivalence=False,require_remote_experts=False,
                         require_remote_decode_equivalence=False)
            report,_=check()
            self.assertEqual(report['remote_expert_layer_ids'],SELECTED_IDS)
            self.assertEqual(report['remote_expert_payload_bytes'],SELECTED_PAYLOAD)
            for key,value in (('remote_expert_layer_ids_for_dual',[22,28,28,34]),
                              ('remote_expert_layer_ids_for_dual',[22,28,30,48]),
                              ('remote_expert_payload_bytes_for_dual',0),
                              ('remote_expert_layer_ids_explicit_for_dual','true'),
                              ('remote_expert_layer_ids_explicit_for_dual',False)):
                previous=gates[key];gates[key]=value
                with self.assertRaises(RuntimeError):check()
                gates[key]=previous

    def test_full_trial_requires_exact_batch_mtp_admission(self):
        gates={**placement_proof(),'passed':True,'executable_sha256':'testexe','source_hashes':{},
               'bound_files':[],'evidence_scope':'unit test'}
        parity={'passed':True,'engine_executable_sha256':'testexe','gates_sha256':'testgates',
                'fixtures':MTP_FIXTURES,'modes':MTP_MODES,'probability_bits':'exact',
                'bound_evidence':[{'path':str(ROOT/'parity-runs/mock01'/name),'sha256':'testgates'}
                                  for name in ('result.json','native.log','probe.json','command.json')],
                'scope':PARITY_SCOPE}
        def read(path,*args,**kwargs):
            return json.dumps(parity if path.name=='mtp-parity-admission.json' else gates)
        with patch.object(shared,'collect_report',return_value={}), \
                patch.object(shared,'EXE_HASH','testexe'), \
                patch.object(shared,'digest',return_value='testgates'), \
                patch.object(advanced_guard,'source_hashes',return_value={}), \
                patch.object(Path,'read_text',read):
            report,_=advanced_guard.artifact_checks('experimental-no-vram-floor',require_remote_experts=False)
            self.assertTrue(report['mtp_layer_equivalence_checked'])
            for key,value in (('probability_bits',None),('probability_bits','tolerance'),
                              ('modes',MTP_MODES-1),('fixtures',MTP_FIXTURES-1),('scope','other'),
                              ('engine_executable_sha256','old'),('gates_sha256','old')):
                previous=parity[key]
                parity[key]=value
                with self.assertRaises(RuntimeError):
                    advanced_guard.artifact_checks('experimental-no-vram-floor',require_remote_experts=False)
                parity[key]=previous
            previous=parity['bound_evidence']
            parity['bound_evidence']=[]
            with self.assertRaises(RuntimeError):
                advanced_guard.artifact_checks('experimental-no-vram-floor',require_remote_experts=False)
            parity['bound_evidence']=previous[:-1]+[previous[0]]
            with self.assertRaises(RuntimeError):
                advanced_guard.artifact_checks('experimental-no-vram-floor',require_remote_experts=False)
            parity['bound_evidence']=previous

    def test_full_trial_requires_exact_resident_expert_admission(self):
        gates={**placement_proof(),'passed':True,'executable_sha256':'testexe','source_hashes':{},'bound_files':[],'evidence_scope':'unit test'}
        parity={'passed':True,'engine_executable_sha256':'testexe','gates_sha256':'testgates',
                'layers':4,'decode_calls':6,'prefill_calls':5,'quant_bytes':'exact','outputs':'exact','blob_bytes':'exact',
                'layer_ids':SELECTED_IDS,'payload_bytes':SELECTED_PAYLOAD,
                'scope':advanced_guard.REMOTE_SCOPE,
                'bound_evidence':[{'path':str(ROOT/'parity-runs/mock02'/name),'sha256':'testgates'}
                                  for name in ('result.json','native.log','probe.json','command.json')]}
        def read(path,*args,**kwargs):
            return json.dumps(parity if path.name=='remote-experts-admission.json' else gates)
        with patch.object(shared,'collect_report',return_value={}),patch.object(shared,'EXE_HASH','testexe'), \
             patch.object(shared,'digest',return_value='testgates'),patch.object(advanced_guard,'source_hashes',return_value={}), \
             patch.object(Path,'read_text',read):
            report,_=advanced_guard.artifact_checks('experimental-no-vram-floor',require_mtp_equivalence=False)
            self.assertTrue(report['remote_expert_equivalence_checked'])
            for key,value in (('layers',3),('decode_calls',5),('prefill_calls',4),('quant_bytes','tolerance'),
                              ('outputs',None),('blob_bytes','tolerance'),('layer_ids',[0,1,2,3]),
                              ('payload_bytes',3250585600),('scope','other'),('engine_executable_sha256','old'),('gates_sha256','old'),
                              ('bound_evidence',[])):
                previous=parity[key]
                parity[key]=value
                with self.assertRaises(RuntimeError):
                    advanced_guard.artifact_checks('experimental-no-vram-floor',require_mtp_equivalence=False)
                parity[key]=previous

    def test_config_pair_matches_the_selected_strategy(self):
        one=json.loads((ROOT/'single-config.json').read_text())
        two=json.loads((ROOT/'dual-config.json').read_text())
        self.assertEqual(one['strategy'],two['strategy'])
        gates=json.loads((ROOT/'source-gates.json').read_text())
        one_args=list(one['args']);two_args=list(two['args'])
        one_ids,one_selected=remote_placement(one_args)
        two_ids,two_selected=remote_placement(two_args)
        self.assertEqual((one_ids,one_selected),([],False))
        expected_ids=gates['remote_expert_layer_ids_for_dual'] if one['strategy']=='resident-experts' else []
        expected_selected=gates['remote_expert_layer_ids_explicit_for_dual'] if expected_ids else False
        self.assertEqual((two_ids,two_selected),(expected_ids,expected_selected))

        def pop_option(argv,flag):
            i=argv.index(flag);return argv.pop(i),argv.pop(i)
        pop_option(one_args,'--remote-expert-layer-ids' if one_selected else '--remote-expert-layers')
        pop_option(two_args,'--remote-expert-layer-ids' if two_selected else '--remote-expert-layers')
        for argv,value in ((one_args,'0'),(two_args,'0' if one['strategy']=='resident-experts' else '1')):
            i=argv.index('--mtp-device');self.assertEqual(argv[i+1],value);argv[i+1]='0'
        for argv,value in ((one_args,'original'),(two_args,gates['remote_decode_mode_for_dual'] if expected_ids else 'original')):
            i=argv.index('--remote-expert-mode');self.assertEqual(argv[i+1],value);argv[i+1]='original'
        advanced_guard.hybrid_placement(one_args,False)
        advanced_guard.frozen_remote_cache_placement(one_args,False)
        if advanced_guard.frozen_remote_cache_placement(two_args,bool(expected_ids) and gates.get('remote_cache_frozen_for_dual',False)):
            two_args.remove('--remote-expert-freeze-cache')
        if advanced_guard.hybrid_placement(two_args,bool(expected_ids) and gates.get('remote_hybrid_for_dual',False)):
            two_args.remove('--remote-expert-hybrid')
        self.assertEqual({**one,'args':one_args},{**two,'args':two_args})
        self.assertEqual(one['args'][one['args'].index('--max-context')+1],'4096')
        slots=int(one['args'][one['args'].index('--expert-cache')+1])
        self.assertIn(slots,(3072,3500,4000,4359,4903,5500))
        self.assertIn('--expert-cache-exact',one['args'])
        self.assertEqual(one['args'][one['args'].index('--expert-pin-mib')+1],'16384')
        self.assertIn('--mtp-resident-embedding',one['args'])
        self.assertIn('--mtp-mapped-residuals',one['args'])
        self.assertEqual('--mtp-chain-batch' in one['args'],gates['matched_mtp_chain_batch'])
        self.assertEqual('--verify-wait-profile' in one['args'],gates['matched_verify_wait_profile'])

    def test_optimized_decode_requires_fresh_bound_mode_admission(self):
        gates={**placement_proof(),'passed':True,'executable_sha256':'testexe','source_hashes':{},'bound_files':[],
               'evidence_scope':'unit test','remote_decode_mode_for_dual':'graphs'}
        parity={'passed':True,'engine_executable_sha256':'testexe','gates_sha256':'testgates',
                'layers':4,'bursts':64,'modes':3,'devices':2,'conditions':24,'outputs':'exact',
                'reference_bytes':65536000,'k4_fallback_exact':True,'graph_replays_per_graph_condition':256,
                'layer_ids':SELECTED_IDS,'payload_bytes':SELECTED_PAYLOAD,
                'scope':advanced_guard.DECODE_SCOPE,
                'bound_evidence':[{'path':str(ROOT/'parity-runs/mock03'/name),'sha256':'testgates'}
                                  for name in ('result.json','native.log','probe.json','command.json')]}
        def read(path,*args,**kwargs):
            return json.dumps(parity if path.name=='remote-decode-admission.json' else gates)
        with patch.object(shared,'collect_report',return_value={}),patch.object(shared,'EXE_HASH','testexe'), \
             patch.object(shared,'digest',return_value='testgates'),patch.object(advanced_guard,'source_hashes',return_value={}), \
             patch.object(Path,'read_text',read):
            def check():
                return advanced_guard.artifact_checks('experimental-no-vram-floor',
                         require_mtp_equivalence=False,require_remote_experts=False)
            report,_=check()
            self.assertTrue(report['remote_decode_equivalence_checked'])
            for key,value in (('modes',2),('conditions',23),('reference_bytes',100),('outputs','tolerance'),
                              ('k4_fallback_exact',False),('graph_replays_per_graph_condition',0),('scope','other'),
                              ('layer_ids',[0,1,2,3]),('payload_bytes',3250585600),
                              ('engine_executable_sha256','old'),('gates_sha256','old'),('bound_evidence',[])):
                previous=parity[key]
                parity[key]=value
                with self.assertRaises(RuntimeError):check()
                parity[key]=previous
            old=parity['bound_evidence'][0]['path']
            parity['bound_evidence'][0]['path']=str(STOCK/'parity-runs/mock03/result.json')
            with self.assertRaises(RuntimeError):check()
            parity['bound_evidence'][0]['path']=old
            report,_=advanced_guard.artifact_checks('experimental-no-vram-floor',require_mtp_equivalence=False,
                     require_remote_experts=False,require_remote_decode_equivalence=False)
            self.assertFalse(report['remote_decode_equivalence_checked'])

    def test_latest_source_is_bound_to_gates(self):
        gates=json.loads((ROOT/'source-gates.json').read_text())
        self.assertTrue(gates['passed'])
        self.assertEqual(source_hashes(),gates['source_hashes'])
        self.assertEqual(gates['transport']['cases'],64)
        self.assertEqual(gates['registration']['device_reads'],6)
        self.assertEqual(gates['matched_expert_pin_mib'],16384)
        self.assertEqual(gates['embedding']['cases'],64)
        self.assertTrue(gates['matched_mtp_resident_embedding'])
        self.assertTrue(gates['matched_mtp_mapped_residuals'])
        self.assertIn(gates['matched_mtp_chain_batch'],(True,False))
        self.assertEqual(gates['compacted_loader']['negative_cases'],4)
        selected=gates['compacted_loader']['selected_host_layout']
        self.assertEqual(selected,{'payload_bytes':80,'layers':3,'scatter':'exact',
                                   'native_roles':'exact','aliases':'exact','negatives':'exact'})
        ids=gates['remote_expert_layer_ids_for_dual']
        self.assertEqual(parse_remote_layer_ids(','.join(map(str,ids))),ids)
        dual=json.loads((ROOT/'dual-config.json').read_text())
        runtime_ids,explicit=remote_placement(dual['args'])
        if dual['strategy']=='resident-experts':
            self.assertEqual(ids,runtime_ids)
            self.assertEqual(gates['remote_expert_layer_ids_explicit_for_dual'],explicit)
        else:
            self.assertEqual((runtime_ids,ids,explicit),([],[0,1,2,3],False))
        self.assertEqual(gates['remote_cli_boundary_cases'],31)
        self.assertEqual(gates['compacted_loader']['selected6_host_layout'],advanced_guard.SIX_HOST_PROOF)
        host_path=Path(gates['compacted_loader']['log'])
        self.assertEqual(host_path.read_text().splitlines().count(advanced_guard.SIX_HOST_MARKER),1)
        self.assertIn(str(host_path),[row['path'] for row in gates['bound_files']])
        pack=Path(dual['args'][dual['args'].index('--pack')+1])
        self.assertEqual(gates['remote_expert_payload_bytes_for_dual'],
                         native_expert_payload(pack,ids))
        self.assertTrue(gates['remote_expert_equivalence_required_for_full_trial'])
        self.assertTrue(gates['mtp_equivalence_required_for_full_trial'])
        self.assertEqual(gates['verify_wait']['devices'],2)
        self.assertEqual(gates['verify_wait']['profiled_replays_per_device'],4)
        self.assertEqual(gates['verify_wait']['outputs'],'exact')
        for row in gates['bound_files']:
            self.assertEqual(shared.digest(Path(row['path'])),row['sha256'])


if __name__=='__main__':
    unittest.main(verbosity=2)
