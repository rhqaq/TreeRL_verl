"""
TreeRL 单元测试 - 不依赖外部库
"""

# ==============================================================================
# TreeNode 测试
# ==============================================================================

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TreeNode:
    """树节点，用于熵引导的树搜索"""
    tree_idx: int
    node_idx: int
    token_ids: List[int]
    log_probs: List[float]
    parent: Optional['TreeNode'] = None
    parent_split_idx: int = 0
    children: List['TreeNode'] = field(default_factory=list)
    mask: List[bool] = field(default_factory=list)
    binary_score: float = 0.0
    
    def __post_init__(self):
        self.mask = [False] * len(self.token_ids)
        if self.parent is not None and len(self.token_ids) > 0:
            self.mask[0] = True
    
    def get_prefix_ids(self, split_idx: int) -> List[int]:
        """获取到 split_idx 为止的前缀 token ids"""
        prefix = []
        if self.parent is not None:
            prefix = self.parent.get_prefix_ids(self.parent_split_idx)
        prefix.extend(self.token_ids[:split_idx])
        return prefix
    
    def get_full_ids(self) -> List[int]:
        """获取完整序列"""
        if self.parent is not None:
            return self.parent.get_full_ids() + self.token_ids
        return self.token_ids.copy()
    
    def get_high_entropy_tokens(self, top_n: int = 2) -> List[int]:
        """获取高熵（低概率）的 token 索引"""
        entropies = [(-log_prob, idx) for idx, log_prob in enumerate(self.log_probs) if not self.mask[idx]]
        entropies.sort(reverse=True)
        return [idx for _, idx in entropies[:top_n]]
    
    def update_mask(self, split_idx: int):
        """更新 mask，标记已扩展的部分"""
        for i in range(min(split_idx, len(self.mask))):
            self.mask[i] = True


def test_treenode():
    print("Testing TreeNode...")
    
    # Test 1: 创建根节点
    root = TreeNode(
        tree_idx=0,
        node_idx=0,
        token_ids=[1, 2, 3, 4, 5],
        log_probs=[-0.1, -0.5, -1.0, -0.2, -0.8]
    )
    
    assert root.tree_idx == 0
    assert root.node_idx == 0
    assert len(root.token_ids) == 5
    assert len(root.mask) == 5
    print("  ✓ Root node creation")
    
    # Test 2: 获取高熵 token
    high_entropy = root.get_high_entropy_tokens(top_n=2)
    # log_probs: [-0.1, -0.5, -1.0, -0.2, -0.8]
    # entropy: [0.1, 0.5, 1.0, 0.2, 0.8]
    # top 2: index 2 (1.0) and index 4 (0.8)
    assert 2 in high_entropy
    assert 4 in high_entropy
    print(f"  ✓ High entropy tokens: {high_entropy}")
    
    # Test 3: 添加子节点
    child = TreeNode(
        tree_idx=0,
        node_idx=1,
        token_ids=[6, 7],
        log_probs=[-0.3, -0.4],
        parent=root,
        parent_split_idx=3
    )
    root.children.append(child)
    
    assert child.parent == root
    assert child.parent_split_idx == 3
    print("  ✓ Child node creation")
    
    # Test 4: 获取前缀
    prefix = child.get_prefix_ids(1)
    # prefix = root.get_prefix_ids(3) + child.token_ids[:1]
    # root 没有父节点，所以 prefix = root.token_ids[:3] = [1, 2, 3]
    # 然后加上 child.token_ids[:1] = [6]
    # 结果应该是 [1, 2, 3, 6]
    assert prefix == [1, 2, 3, 6], f"Expected [1,2,3,6], got {prefix}"
    print(f"  ✓ Prefix extraction: {prefix}")
    
    # Test 5: 获取完整序列
    full_ids = child.get_full_ids()
    assert full_ids == [1, 2, 3, 4, 5, 6, 7], f"Expected [1,2,3,4,5,6,7], got {full_ids}"
    print(f"  ✓ Full sequence: {full_ids}")
    
    # Test 6: 更新 mask
    root.update_mask(2)
    assert root.mask[0] == True
    assert root.mask[1] == True
    print("  ✓ Mask update")
    
    # Test 7: 高熵 token 排除已 mask 的
    high_entropy_after_mask = root.get_high_entropy_tokens(top_n=2)
    # 已 mask: index 0, 1
    # 可选: index 2 (entropy=1.0), 3 (entropy=0.2), 4 (entropy=0.8)
    # top 2: index 2, 4
    assert 2 in high_entropy_after_mask
    assert 4 in high_entropy_after_mask
    print(f"  ✓ High entropy after mask: {high_entropy_after_mask}")
    
    print("\n✓ All TreeNode tests passed!")


def test_tree_structure():
    print("\nTesting tree structure...")
    
    # 构建一个复杂的树
    #       root
    #      /    \
    #   child1  child2
    #    |
    #  grandchild
    
    root = TreeNode(0, 0, [1, 2, 3], [-0.5, -1.0, -0.3])
    child1 = TreeNode(0, 1, [4, 5], [-0.2, -0.6], parent=root, parent_split_idx=2)
    child2 = TreeNode(0, 2, [6], [-0.4], parent=root, parent_split_idx=1)
    grandchild = TreeNode(0, 3, [7, 8], [-0.7, -0.1], parent=child1, parent_split_idx=1)
    
    root.children = [child1, child2]
    child1.children = [grandchild]
    
    # 测试完整序列
    assert root.get_full_ids() == [1, 2, 3]
    assert child1.get_full_ids() == [1, 2, 3, 4, 5]
    assert child2.get_full_ids() == [1, 2, 3, 6]
    assert grandchild.get_full_ids() == [1, 2, 3, 4, 5, 7, 8], f"Got {grandchild.get_full_ids()}"
    print("  ✓ Full sequences correct")
    
    # 测试前缀
    # grandchild 的 parent_split_idx=1，表示 child1 在位置 1 被分割
    # grandchild.get_prefix_ids(0) = child1.get_prefix_ids(1) + grandchild.token_ids[:0]
    # child1.get_prefix_ids(1) = root.get_prefix_ids(2) + child1.token_ids[:1]
    # root.get_prefix_ids(2) = root.token_ids[:2] = [1, 2]
    # child1.token_ids[:1] = [4]
    # 所以结果是 [1, 2, 4]
    assert grandchild.get_prefix_ids(0) == [1, 2, 4], f"Got {grandchild.get_prefix_ids(0)}"
    print("  ✓ Prefix correct")
    
    print("\n✓ All tree structure tests passed!")


if __name__ == "__main__":
    test_treenode()
    test_tree_structure()
    print("\n" + "="*50)
    print("All tests passed successfully!")
    print("="*50)
