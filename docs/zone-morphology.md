# Zone 形态区分与正典依据

本轮承接用户对 `generated/zone-evidence/profiles.png` 的肉眼验收：区域生成、
过渡和六种鲜明地貌已得到认可，继续拆开同心环／阶地与低起伏杂斑两组相似形态。
验收图仍共用 45–310 高程色标；目标是空间结构可辨认。

## 正典来源

本仓库没有 `docs/worldview.md`。读取的是相邻主仓库
`/home/serverkizuna/Code/Bong/docs/worldview.md`，Bong HEAD 为
`cce2b6a78d42b919a98902b683930088d9388589`，文件 SHA-256 为
`6602eeb5cbd89bee3aae5046dcf3ca02f4715a841142bab636a55c4d9672f726`。
下列行号均对应该版本，引用事实与地貌实现选择分开记录。

| 对象 | 正典约束 | 本轮形态选择 |
|---|---|---|
| 渡劫焦土 | worldview.md §八 L787：天劫降于修士；worldview.md §十七 L1688：焦地保留永久伤痕 | 一个局部灼击凹心、断开的隆起段和向外分叉的雷痕沟 |
| 裂口荒原 | worldview.md §十六 L1409：普通地壳裂缝、洞穴与新鲜崩石可为入口 | 纵贯的折线裂口、旁支塌陷、不等长错位崖肩，两端敞开 |
| 宗门遗迹 | worldview.md §一 L15：残破洞府与失控禁制；worldview.md §十七 L1686：遗迹／阵核 | 分离的方形台基、残留院落边界及断口 |
| 丹宗遗园 | 同上；worldview.md §十 L886：灵草用于炼丹；本库 POI「百草丹殿」标记 alchemy | 有公共中轴的三列高程园圃，平行长条与间隔小路 |
| 王印台 | 通用遗迹约束同上；本库 POI「观天台」标记 vortex_formation | 集中的菱形高台和轴向踏级，与散落院落／条带园圃区分 |
| 战魂平野 | worldview.md §十六 L1386：战亡者聚集与真元沉淀；worldview.md §十六 L1471：多为破损品 | 多处局部冲击凹地、交错长沟、遗迹残基与冢脊；依据本库破碎阵眼、万骨冢等 POI 布局 |
| 幽暗地穴 | worldview.md §十三 L1267：地下网络；worldview.md §十六 L1409：洞穴／崩石入口 | 地表保留覆盖岩层，以分散陷穴和入口周边的塌陷沟提示地下 |
| 无垠深渊 | 正典没有同名固定区域的细节；本库 POI 有三层禁殿与坠渊井 | 依照现有入口与井的锚点表达集中裂隙和阶状崩缘，保留已有三层地下网络 |

正典没有规定这些遗迹的平面几何、台阶尺寸或雷痕分叉角；这些属于可调整的地貌
实现参数，不作为新增世界观事实。`worldview.md §十六 L1377` 明确活坍缩渊是
独立位面，主世界裂缝只提供入口锚点。因此地表形态不承担秘境内部生成，也不把
本库正灵压的固定 `wuxing_abyss` 自动解释为该类临时秘境。

北荒 `north_wastes` 的 `boundary_mode=hard`、`boundary_width=96` 按既有规则
使用 48 格连续过渡；平坦台地与背景的高度差使边缘锐利，是数据选择。

## 实施与验收

按灼击／地裂、人造遗迹、战场、地下区域地表线索分提交。上层将几何锚点转为世界
坐标，engine 只解释通用高度场参数。保留原有 148 项断言，新增横断面、开口、
结构方向与断裂等空间契约，三个 seed 验证形态不靠偶然噪声成立。

原验收图与采样已留在 `generated/zone-evidence-before-morphology/`。
最终重跑 `tools/zone_evidence.py` 和 `tools/plot_zone_evidence.py`，由用户肉眼验收
`generated/zone-evidence/profiles.png`。所有统计只作为可复现辅助，不替代看图。

## 用户暂停时的断点

灼击／地裂、人造遗迹和战场三组形态已实现，完整回归为 198 项通过。用户要求暂停，
本轮尚未重新生成验收图；`generated/zone-evidence/profiles.png` 仍是此前的版本，
不能用于验收这三组新形态。

恢复后先处理 `cave_network` / `abyssal_maze` 的地表线索，保留 `ash_dead_zone`。
表中的地下形态是待实现方向，当前配方尚未修改。洞穴高度跟随局部地表，调整时需
保护既有三层空气／岩层、POI 支撑面和井口连通契约，并重新运行地下连通证据扫描；
尤其保留 `tests/test_zone_caves.py` 中三层壁片回归的原断言。

完成这组后重新采样全部区域、生成 `generated/zone-evidence/profiles.png`，
向用户提供路径进行肉眼验收，并提交可复现证据。基线仍保留在
`generated/zone-evidence-before-morphology/`。不启动或排查 BlueMap serve。
