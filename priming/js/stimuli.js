/// init ///
let allStimuli = [

    // Phon-sem:
    {'prime': '渡', 'target': '量', 'trial_type': 'phon-sem'},
    {'prime': '佑', 'target': '左', 'trial_type': 'phon-sem'},
    {'prime': '誌', 'target': '願', 'trial_type': 'phon-sem'},
    {'prime': '紛', 'target': '析', 'trial_type': 'phon-sem'},
    {'prime': '泥', 'target': '姑', 'trial_type': 'phon-sem'},
    {'prime': '楓', 'target': '雨', 'trial_type': 'phon-sem'},
    {'prime': '餵', 'target': '懼', 'trial_type': 'phon-sem'},
    {'prime': '喧', 'target': '佈', 'trial_type': 'phon-sem'},
    {'prime': '財', 'target': '能', 'trial_type': 'phon-sem'},
    {'prime': '傢', 'target': '人', 'trial_type': 'phon-sem'},
    {'prime': '儘', 'target': '量', 'trial_type': 'phon-sem'},
    {'prime': '胸', 'target': '奴', 'trial_type': 'phon-sem'},
    {'prime': '謂', 'target': '腸', 'trial_type': 'phon-sem'},
    {'prime': '錶', 'target': '示', 'trial_type': 'phon-sem'},
    {'prime': '附', 'target': '出', 'trial_type': 'phon-sem'},
    {'prime': '濃', 'target': '田', 'trial_type': 'phon-sem'},
    {'prime': '据', 'target': '住', 'trial_type': 'phon-sem'},
    {'prime': '棋', 'target': '實', 'trial_type': 'phon-sem'},
    {'prime': '勵', 'target': '害', 'trial_type': 'phon-sem'},
    {'prime': '婚', 'target': '迷', 'trial_type': 'phon-sem'},
    {'prime': '健', 'target': '立', 'trial_type': 'phon-sem'},
    {'prime': '拒', 'target': '大', 'trial_type': 'phon-sem'},
    {'prime': '燈', 'target': '山', 'trial_type': 'phon-sem'},
    {'prime': '楣', 'target': '毛', 'trial_type': 'phon-sem'},
    {'prime': '烤', 'target': '試', 'trial_type': 'phon-sem'},
    {'prime': '締', 'target': '國', 'trial_type': 'phon-sem'},
    {'prime': '擒', 'target': '獸', 'trial_type': 'phon-sem'},
    {'prime': '億', 'target': '義', 'trial_type': 'phon-sem'},
    {'prime': '溶', 'target': '易', 'trial_type': 'phon-sem'},
    {'prime': '讚', 'target': '成', 'trial_type': 'phon-sem'},
    {'prime': '銘', 'target': '字', 'trial_type': 'phon-sem'},
    {'prime': '綜', 'target': '教', 'trial_type': 'phon-sem'},
    {'prime': '壢', 'target': '史', 'trial_type': 'phon-sem'},
    {'prime': '仗', 'target': '夫', 'trial_type': 'phon-sem'},
    {'prime': '汐', 'target': '陽', 'trial_type': 'phon-sem'},
    {'prime': '城', 'target': '功', 'trial_type': 'phon-sem'},
    {'prime': '程', 'target': '現', 'trial_type': 'phon-sem'},
    {'prime': '惶', 'target': '帝', 'trial_type': 'phon-sem'},
    {'prime': '湧', 'target': '敢', 'trial_type': 'phon-sem'},
    {'prime': '源', 'target': '因', 'trial_type': 'phon-sem'},
    {'prime': '遵', 'target': '重', 'trial_type': 'phon-sem'},
    {'prime': '泳', 'target': '遠', 'trial_type': 'phon-sem'},
    {'prime': '倖', 'target': '福', 'trial_type': 'phon-sem'},
    {'prime': '稀', 'target': '望', 'trial_type': 'phon-sem'},
    {'prime': '棚', 'target': '友', 'trial_type': 'phon-sem'},

    // Sem-sem:
    {'prime': '孝', 'target': '順', 'trial_type': 'sem-sem'},
    {'prime': '剪', 'target': '刀', 'trial_type': 'sem-sem'},
    {'prime': '畢', 'target': '業', 'trial_type': 'sem-sem'},
    {'prime': '搔', 'target': '癢', 'trial_type': 'sem-sem'},
    {'prime': '寫', 'target': '字', 'trial_type': 'sem-sem'},
    {'prime': '諷', 'target': '刺', 'trial_type': 'sem-sem'},
    {'prime': '仇', 'target': '恨', 'trial_type': 'sem-sem'},
    {'prime': '冰', 'target': '冷', 'trial_type': 'sem-sem'},
    {'prime': '虐', 'target': '待', 'trial_type': 'sem-sem'},
    {'prime': '囚', 'target': '犯', 'trial_type': 'sem-sem'},
    {'prime': '拯', 'target': '救', 'trial_type': 'sem-sem'},
    {'prime': '革', 'target': '命', 'trial_type': 'sem-sem'},
    {'prime': '緊', 'target': '張', 'trial_type': 'sem-sem'},
    {'prime': '奴', 'target': '隸', 'trial_type': 'sem-sem'},
    {'prime': '黑', 'target': '白', 'trial_type': 'sem-sem'},
    {'prime': '盔', 'target': '甲', 'trial_type': 'sem-sem'},
    {'prime': '普', 'target': '通', 'trial_type': 'sem-sem'},
    {'prime': '審', 'target': '判', 'trial_type': 'sem-sem'},
    {'prime': '稻', 'target': '米', 'trial_type': 'sem-sem'},
    {'prime': '輸', 'target': '贏', 'trial_type': 'sem-sem'},
    {'prime': '克', 'target': '服', 'trial_type': 'sem-sem'},
    {'prime': '手', 'target': '腳', 'trial_type': 'sem-sem'},
    {'prime': '迅', 'target': '速', 'trial_type': 'sem-sem'},
    {'prime': '森', 'target': '林', 'trial_type': 'sem-sem'},
    {'prime': '饑', 'target': '餓', 'trial_type': 'sem-sem'},
    {'prime': '划', 'target': '船', 'trial_type': 'sem-sem'},
    {'prime': '針', 'target': '線', 'trial_type': 'sem-sem'},
    {'prime': '羨', 'target': '慕', 'trial_type': 'sem-sem'},
    {'prime': '厭', 'target': '惡', 'trial_type': 'sem-sem'},
    {'prime': '簡', 'target': '單', 'trial_type': 'sem-sem'},
    {'prime': '邏', 'target': '輯', 'trial_type': 'sem-sem'},
    {'prime': '氾', 'target': '濫', 'trial_type': 'sem-sem'},
    {'prime': '東', 'target': '西', 'trial_type': 'sem-sem'},
    {'prime': '肥', 'target': '胖', 'trial_type': 'sem-sem'},
    {'prime': '真', 'target': '實', 'trial_type': 'sem-sem'},
    {'prime': '蔬', 'target': '菜', 'trial_type': 'sem-sem'},
    {'prime': '離', 'target': '開', 'trial_type': 'sem-sem'},
    {'prime': '早', 'target': '晚', 'trial_type': 'sem-sem'},
    {'prime': '喜', 'target': '歡', 'trial_type': 'sem-sem'},
    {'prime': '廁', 'target': '所', 'trial_type': 'sem-sem'},
    {'prime': '嬰', 'target': '兒', 'trial_type': 'sem-sem'},
    {'prime': '尷', 'target': '尬', 'trial_type': 'sem-sem'},
    {'prime': '危', 'target': '險', 'trial_type': 'sem-sem'},
    {'prime': '垃', 'target': '圾', 'trial_type': 'sem-sem'},
    {'prime': '夫', 'target': '妻', 'trial_type': 'sem-sem'},

    //Control:
    {'prime': '很', 'target': '車', 'trial_type': 'con'},
    {'prime': '帳', 'target': '煮', 'trial_type': 'con'},
    {'prime': '余', 'target': '別', 'trial_type': 'con'},
    {'prime': '擁', 'target': '賞', 'trial_type': 'con'},
    {'prime': '糖', 'target': '鄰', 'trial_type': 'con'},
    {'prime': '峽', 'target': '述', 'trial_type': 'con'},
    {'prime': '極', 'target': '損', 'trial_type': 'con'},
    {'prime': '藥', 'target': '邀', 'trial_type': 'con'},
    {'prime': '宜', 'target': '其', 'trial_type': 'con'},
    {'prime': '秩', 'target': '肯', 'trial_type': 'con'},
    {'prime': '銀', 'target': '蓋', 'trial_type': 'con'},
    {'prime': '軸', 'target': '階', 'trial_type': 'con'},
    {'prime': '漆', 'target': '端', 'trial_type': 'con'},
    {'prime': '梓', 'target': '某', 'trial_type': 'con'},
    {'prime': '額', 'target': '賽', 'trial_type': 'con'},
    {'prime': '叫', 'target': '六', 'trial_type': 'con'},
    {'prime': '村', 'target': '即', 'trial_type': 'con'},
    {'prime': '形', 'target': '再', 'trial_type': 'con'},
    {'prime': '瑰', 'target': '殺', 'trial_type': 'con'},
    {'prime': '跟', 'target': '董', 'trial_type': 'con'},
    {'prime': '孩', 'target': '侵', 'trial_type': 'con'},
    {'prime': '準', 'target': '腦', 'trial_type': 'con'},
    {'prime': '司', 'target': '進', 'trial_type': 'con'},
    {'prime': '移', 'target': '勝', 'trial_type': 'con'},
    {'prime': '漢', 'target': '隔', 'trial_type': 'con'},
    {'prime': '根', 'target': '候', 'trial_type': 'con'},
    {'prime': '眠', 'target': '務', 'trial_type': 'con'},
    {'prime': '臭', 'target': '堂', 'trial_type': 'con'},
    {'prime': '檔', 'target': '諾', 'trial_type': 'con'},
    {'prime': '煉', 'target': '葉', 'trial_type': 'con'},
    {'prime': '炸', 'target': '查', 'trial_type': 'con'},
    {'prime': '殖', 'target': '游', 'trial_type': 'con'},
    {'prime': '樣', 'target': '輕', 'trial_type': 'con'},
    {'prime': '盪', 'target': '棍', 'trial_type': 'con'},
    {'prime': '首', 'target': '俊', 'trial_type': 'con'},
    {'prime': '邦', 'target': '鎖', 'trial_type': 'con'},
    {'prime': '繪', 'target': '港', 'trial_type': 'con'},
    {'prime': '那', 'target': '飛', 'trial_type': 'con'},
    {'prime': '遂', 'target': '喝', 'trial_type': 'con'},
    {'prime': '擅', 'target': '廉', 'trial_type': 'con'},
    {'prime': '壯', 'target': '另', 'trial_type': 'con'},
    {'prime': '戚', 'target': '唐', 'trial_type': 'con'},
    {'prime': '企', 'target': '天', 'trial_type': 'con'},
    {'prime': '雖', 'target': '營', 'trial_type': 'con'},
    {'prime': '黎', 'target': '彬', 'trial_type': 'con'}
];

  
  //sessionStorage.setItem("allStimuli");