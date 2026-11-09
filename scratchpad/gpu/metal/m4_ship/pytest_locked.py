"""Run all pytest tests; adapt only the new Vulkan fixture's parent-lock naming."""
import os,sys
from pathlib import Path
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tests'))
import test_vulkan
original=test_vulkan._acquire_lock
expected=os.environ['LMZ_PARENT_LOCK_TOKEN']
def acquire():
 owner=test_vulkan.LOCK_DIR/'owner'
 if owner.is_file() and owner.read_text()==expected:return
 original()
test_vulkan._acquire_lock=acquire
import pytest
raise SystemExit(pytest.main(['-q','tests']))
