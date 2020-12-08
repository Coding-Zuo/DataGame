用户特征提取：  

| 特征 | 数据类型 | 说明 |
| :--- | :--- | :--- |
|一阶特征 10个|
| user_questions_count | int32 | 用户历史做题量 |
| user_correctCount | int32 | 用户历史回答正确的数量|
| user_explanation_count | int16 | 用户查看问题答案次数 |
| user_thisQuestion_count | int32 | 用户对当前问题id 的历史统计 |
| user_thisQuestion_correctCount | int32 | 用户对当前问题id 的历史正确数 |
| user_thisPart_count | int16 | 用户当前Part 统计 |
| user_thisPart_correctCount | int16 | 用户当前Part 正确统计 |
| user_thisTags_count | int16 | 用户当前count 统计 |
| user_thisTags_correctCount | int16 | 用户当前count 正确统计 |
| user_elapsed_time_categorical | int16 | 用户做题时间，按秒计算大于300秒记为300秒 |
| user_lag_time_categorical | int16 | （**暂无**）用户当前题与上一道题的时间间隔，按0, 1, 2, 3, 4, 5, 10, 20, 30, . . . , 1440.分钟分箱 |
|二阶特征 5个|
| user_acc | float32 | 用户历史正确率 |
| user_explanation_rate | float32 | 用户查看问题答案频率 |
| user_thisQuestion_acc | float32 | 用户对当前问题id 的历史正确率(~~弃用，state内存占用过大~~) |
| user_thisPart_acc | float32 | 用户当前Part acc |
| user_thisTags_acc | float32 | 用户当前tags acc |


用户状态表：多重字典

| 状态特征 |数据类型 |说明 |
|:---    |:----  |:---  |
|questions_count |uint32 | 历史做题数量 |
|correctCount |uint32 | 历史做对的题 |
|explanation_count | int16 | 用户查看答案次数 |
|thisQuestion_count | dict |用户当前part 历史统计 |
|thisQuestion_correctCount | dict |用户当前part 历史正确数 |
|thisPart_count | dict |用户当前part 历史统计 |
|thisPart_correctCount | dict |用户当前part 历史正确数 |
|thisTags_count | dict | 当前tags 历史统计|
|thisTags_correctCount | dict |当前tags 历史正确数 | 
|~~user_acc~~|float64 |用户历史正确率 |
|~~explanation_rate~~|float64| 用户历史查看问题答案频率 |
|~~thisQuestion_acc~~ | dict |用户当前part 历史正确率 |
|~~thisPart_acc~~ | dict |用户当前part 历史正确率 |
|~~thisTags_acc~~ | dict |当前tags 历史统计 |

questions (题库)特征提取：  

|  特征   | 数据类型  | 说明   |
|  :----  |:----  | :---- |
| question_id | int32 | 问题 id |
| question_acc | int8 | 问题准确率,取到百分位。eg 85%|
| question_part | int8 | 问题类型 |
| question_part_acc| int8 |part准确率，取到百分位|
| question_tag1 | int16 | 问题的第一个tag |
| question_tag1_acc | int8 | tag1的平均正确率（有待考虑） |
| question_tags_acc | int8 | 问题所有tag的平均正确率 |
| question_tags_community | int8 | 来自[[2020-R3ID] Clustering question tags](https://www.kaggle.com/spacelx/2020-r3id-clustering-question-tags) 的特征,主要是对tags的聚类|
| question_tags_community_acc | int8 |question_tags_community的正确率 |


auc 验证  
|序号| 特征 |数据量:cv/lb |特征重要性|
|:---|:---|:---|:--- |
|1|一阶特征|100w cv:0.755|  |
|2|一阶+二阶|100w cv:0.754| |