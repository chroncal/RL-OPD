"""Scientific gate, shifted prediction and coordinate mask checks; CPU only."""
import json
import unittest
from types import SimpleNamespace
import torch
from .consensus_objectives import consensus_target,coordinate_ce


def candidates(boxes):
    return [{'pct':p,'text':json.dumps({'bbox_2d':b})} for p,b in zip(range(10,101,10),boxes)]


class ConsensusChecks(unittest.TestCase):
    def test_midrange_changed_and_stable_gates(self):
        a=[0,0,100,100];b=[500,500,600,600]
        result=consensus_target(candidates([a]*6+[b]*4))
        self.assertTrue(result['active']);self.assertEqual(result['teacher_bbox'],a)
        self.assertAlmostEqual(result['stability'],21/45)
        self.assertFalse(consensus_target(candidates([a]*10))['active'])
        self.assertTrue(consensus_target(candidates([a]*10),'lower_only')['active'])
        self.assertFalse(consensus_target(candidates([[i*100,0,i*100+50,50] for i in range(10)]))['active'])

    def test_shifted_logits_and_coordinate_only_gradients(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.logits=torch.nn.Parameter(torch.zeros(1,9,11))
            def forward(self,**kwargs):
                assert kwargs['logits_to_keep']==9
                return SimpleNamespace(logits=self.logits)
        model=Toy();mask=[1,4,6]
        ce=coordinate_ce(model,{},list(range(8)),mask);ce.backward()
        self.assertGreater(float(ce),0)
        actual=model.logits.grad.abs().sum((0,2)).nonzero().flatten().tolist()
        self.assertEqual(actual,mask)
        for j in mask:self.assertLess(float(model.logits.grad[0,j,j]),0)


if __name__=='__main__':unittest.main()
