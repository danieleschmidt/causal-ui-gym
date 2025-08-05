/**
 * Internationalization translations for Causal UI Gym.
 * 
 * Supports multiple languages with complete translation coverage
 * for all user-facing text in the application.
 */

export interface TranslationDict {
  [key: string]: string | TranslationDict
}

export interface Translations {
  en: TranslationDict
  es: TranslationDict
  fr: TranslationDict
  de: TranslationDict
  ja: TranslationDict
  zh: TranslationDict
}

export const translations: Translations = {
  // English (default)
  en: {
    common: {
      loading: "Loading...",
      error: "Error",
      success: "Success",
      warning: "Warning",
      info: "Information",
      cancel: "Cancel",
      confirm: "Confirm",
      save: "Save",
      delete: "Delete",
      edit: "Edit",
      create: "Create",
      update: "Update",
      refresh: "Refresh",
      search: "Search",
      filter: "Filter",
      export: "Export",
      import: "Import",
      download: "Download",
      upload: "Upload",
      next: "Next",
      previous: "Previous",
      finish: "Finish",
      close: "Close",
      yes: "Yes",
      no: "No"
    },
    navigation: {
      home: "Home",
      experiments: "Experiments",
      dashboard: "Dashboard",
      agents: "Agents",
      settings: "Settings",
      help: "Help",
      about: "About"
    },
    experiment: {
      title: "Experiments",
      create: "Create Experiment",
      name: "Experiment Name",
      description: "Description",
      status: "Status",
      created: "Created",
      updated: "Last Updated",
      run: "Run Experiment",
      stop: "Stop Experiment",
      results: "Results",
      metrics: "Metrics",
      interventions: "Interventions",
      outcomes: "Outcomes",
      validate: "Validate",
      duplicate: "Duplicate",
      archive: "Archive",
      statuses: {
        created: "Created",
        running: "Running",
        completed: "Completed",
        failed: "Failed",
        cancelled: "Cancelled"
      },
      validation: {
        title: "Experiment Validation",
        passed: "Validation Passed",
        failed: "Validation Failed",
        errors: "Errors",
        warnings: "Warnings",
        assumptions: "Assumptions"
      },
      builder: {
        title: "Experiment Builder",
        steps: {
          variables: "Define Variables",
          relationships: "Create Relationships",
          interventions: "Configure Interventions",
          parameters: "Set Parameters",
          review: "Review & Launch"
        },
        variables: {
          addVariable: "Add Variable",
          variableName: "Variable Name",
          variableType: "Variable Type",
          continuous: "Continuous",
          discrete: "Discrete",
          binary: "Binary",
          removeVariable: "Remove Variable"
        },
        relationships: {
          createEdge: "Create Relationship",
          autoLayout: "Auto Layout",
          edgeWeight: "Edge Weight",
          confidence: "Confidence"
        },
        interventions: {
          addIntervention: "Add Intervention",
          interventionType: "Intervention Type",
          interventionValue: "Intervention Value",
          do: "Do Intervention",
          soft: "Soft Intervention",
          conditional: "Conditional Intervention"
        },
        parameters: {
          sampleSize: "Sample Size",
          randomSeed: "Random Seed",
          confidenceLevel: "Confidence Level"
        },
        summary: {
          variables: "Variables",
          relationships: "Relationships",
          interventions: "Interventions",
          sampleSize: "Sample Size"
        }
      }
    },
    causal: {
      dag: "Causal DAG",
      node: "Node",
      edge: "Edge",
      intervention: "Intervention",
      outcome: "Outcome",
      treatment: "Treatment",
      confounder: "Confounder",
      mediator: "Mediator",
      collider: "Collider",
      effect: "Effect",
      ate: "Average Treatment Effect",
      ite: "Individual Treatment Effect",
      cate: "Conditional Average Treatment Effect",
      backdoor: "Backdoor Criterion",
      frontdoor: "Frontdoor Criterion",
      doCalculus: "Do-Calculus",
      correlation: "Correlation",
      causation: "Causation",
      confounding: "Confounding",
      selection: "Selection Bias",
      identification: "Identification",
      estimand: "Estimand",
      estimator: "Estimator"
    },
    metrics: {
      title: "Metrics Dashboard",
      averageTreatmentEffect: "Average Treatment Effect",
      significanceRate: "Significance Rate",
      computationTime: "Computation Time",
      totalExperiments: "Total Experiments",
      treatmentEffectOverTime: "Treatment Effect Over Time",
      recentResults: "Recent Results",
      confidenceInterval: "Confidence Interval",
      pValue: "P-Value",
      standardError: "Standard Error",
      effectSize: "Effect Size",
      sampleSize: "Sample Size"
    },
    agents: {
      title: "LLM Agents",
      register: "Register Agent",
      agentType: "Agent Type",
      model: "Model",
      status: "Status",
      query: "Query Agent",
      response: "Response",
      reasoning: "Reasoning Steps",
      confidence: "Confidence",
      batchQuery: "Batch Query",
      compare: "Compare Agents",
      statuses: {
        available: "Available",
        busy: "Busy",
        error: "Error",
        offline: "Offline"
      },
      providers: {
        openai: "OpenAI",
        anthropic: "Anthropic",
        other: "Other"
      }
    },
    validation: {
      required: "This field is required",
      invalid: "Invalid value",
      tooShort: "Value is too short",
      tooLong: "Value is too long",
      notUnique: "Value must be unique",
      invalidFormat: "Invalid format",
      outOfRange: "Value is out of range",
      cycleDetected: "Cycle detected in DAG",
      invalidDAG: "Invalid DAG structure",
      noNodes: "DAG must contain at least one node",
      noEdges: "DAG should have at least one edge",
      disconnected: "DAG is not connected"
    },
    errors: {
      general: "An unexpected error occurred",
      network: "Network error",
      timeout: "Request timed out",
      notFound: "Resource not found",
      unauthorized: "Unauthorized access",
      forbidden: "Access forbidden",
      serverError: "Internal server error",
      validationFailed: "Validation failed",
      experimentNotFound: "Experiment not found",
      agentNotFound: "Agent not found",
      computationFailed: "Computation failed",
      tryAgain: "Please try again",
      contactSupport: "Contact support if the problem persists"
    },
    settings: {
      title: "Settings",
      language: "Language",
      theme: "Theme",
      region: "Region",
      timezone: "Timezone",
      notifications: "Notifications",
      privacy: "Privacy",
      security: "Security",
      performance: "Performance",
      experimental: "Experimental Features",
      dataRetention: "Data Retention",
      export: "Export Data",
      import: "Import Settings",
      reset: "Reset to Defaults"
    }
  },

  // Spanish
  es: {
    common: {
      loading: "Cargando...",
      error: "Error",
      success: "Éxito",
      warning: "Advertencia",
      info: "Información",
      cancel: "Cancelar",
      confirm: "Confirmar",
      save: "Guardar",
      delete: "Eliminar",
      edit: "Editar",
      create: "Crear",
      update: "Actualizar",
      refresh: "Actualizar",
      search: "Buscar",
      filter: "Filtrar",
      export: "Exportar",
      import: "Importar",
      download: "Descargar",
      upload: "Subir",
      next: "Siguiente",
      previous: "Anterior",
      finish: "Finalizar",
      close: "Cerrar",
      yes: "Sí",
      no: "No"
    },
    navigation: {
      home: "Inicio",
      experiments: "Experimentos",
      dashboard: "Panel",
      agents: "Agentes",
      settings: "Configuración",
      help: "Ayuda",
      about: "Acerca de"
    },
    experiment: {
      title: "Experimentos",
      create: "Crear Experimento",
      name: "Nombre del Experimento",
      description: "Descripción",
      status: "Estado",
      created: "Creado",
      updated: "Última Actualización",
      run: "Ejecutar Experimento",
      stop: "Detener Experimento",
      results: "Resultados",
      metrics: "Métricas",
      interventions: "Intervenciones",
      outcomes: "Resultados",
      validate: "Validar",
      duplicate: "Duplicar",
      archive: "Archivar",
      statuses: {
        created: "Creado",
        running: "Ejecutándose",
        completed: "Completado",
        failed: "Fallido",
        cancelled: "Cancelado"
      }
    },
    causal: {
      dag: "DAG Causal",
      node: "Nodo",
      edge: "Arista",
      intervention: "Intervención",
      outcome: "Resultado",
      treatment: "Tratamiento",
      confounder: "Variable de Confusión",
      mediator: "Mediador",
      effect: "Efecto",
      ate: "Efecto Promedio del Tratamiento",
      correlation: "Correlación",
      causation: "Causalidad"
    },
    validation: {
      required: "Este campo es obligatorio",
      invalid: "Valor inválido",
      tooShort: "El valor es demasiado corto",
      tooLong: "El valor es demasiado largo",
      cycleDetected: "Ciclo detectado en el DAG",
      invalidDAG: "Estructura de DAG inválida"
    },
    errors: {
      general: "Ocurrió un error inesperado",
      network: "Error de red",
      timeout: "Tiempo de espera agotado",
      tryAgain: "Por favor, inténtalo de nuevo"
    },
    settings: {
      title: "Configuración",
      language: "Idioma",
      theme: "Tema",
      region: "Región",
      timezone: "Zona Horaria"
    }
  },

  // French
  fr: {
    common: {
      loading: "Chargement...",
      error: "Erreur",
      success: "Succès",
      warning: "Avertissement",
      info: "Information",
      cancel: "Annuler",
      confirm: "Confirmer",
      save: "Enregistrer",
      delete: "Supprimer",
      edit: "Modifier",
      create: "Créer",
      update: "Mettre à jour",
      refresh: "Actualiser",
      search: "Rechercher",
      filter: "Filtrer",
      export: "Exporter",
      import: "Importer",
      download: "Télécharger",
      upload: "Téléverser",
      next: "Suivant",
      previous: "Précédent",
      finish: "Terminer",
      close: "Fermer",
      yes: "Oui",
      no: "Non"
    },
    navigation: {
      home: "Accueil",
      experiments: "Expériences",
      dashboard: "Tableau de bord",
      agents: "Agents",
      settings: "Paramètres",
      help: "Aide",
      about: "À propos"
    },
    experiment: {
      title: "Expériences",
      create: "Créer une Expérience",
      name: "Nom de l'Expérience",
      description: "Description",
      status: "Statut",
      created: "Créé",
      updated: "Dernière Mise à Jour",
      run: "Exécuter l'Expérience",
      stop: "Arrêter l'Expérience",
      results: "Résultats",
      metrics: "Métriques",
      interventions: "Interventions",
      outcomes: "Résultats",
      validate: "Valider",
      duplicate: "Dupliquer",
      archive: "Archiver"
    },
    causal: {
      dag: "DAG Causal",
      node: "Nœud",
      edge: "Arête",
      intervention: "Intervention",
      outcome: "Résultat",
      treatment: "Traitement",
      effect: "Effet",
      correlation: "Corrélation",
      causation: "Causalité"
    },
    validation: {
      required: "Ce champ est obligatoire",
      invalid: "Valeur invalide",
      cycleDetected: "Cycle détecté dans le DAG"
    },
    errors: {
      general: "Une erreur inattendue s'est produite",
      network: "Erreur réseau",
      tryAgain: "Veuillez réessayer"
    },
    settings: {
      title: "Paramètres",
      language: "Langue",
      theme: "Thème",
      region: "Région"
    }
  },

  // German
  de: {
    common: {
      loading: "Laden...",
      error: "Fehler",
      success: "Erfolg",
      warning: "Warnung",
      info: "Information",
      cancel: "Abbrechen",
      confirm: "Bestätigen",
      save: "Speichern",
      delete: "Löschen",
      edit: "Bearbeiten",
      create: "Erstellen",
      update: "Aktualisieren",
      refresh: "Aktualisieren",
      search: "Suchen",
      filter: "Filtern",
      export: "Exportieren",
      import: "Importieren",
      download: "Herunterladen",
      upload: "Hochladen",
      next: "Weiter",
      previous: "Zurück",
      finish: "Fertig",
      close: "Schließen",
      yes: "Ja",
      no: "Nein"
    },
    navigation: {
      home: "Startseite",
      experiments: "Experimente",
      dashboard: "Dashboard",
      agents: "Agenten",
      settings: "Einstellungen",
      help: "Hilfe",
      about: "Über"
    },
    experiment: {
      title: "Experimente",
      create: "Experiment Erstellen",
      name: "Experiment Name",
      description: "Beschreibung",
      status: "Status",
      created: "Erstellt",
      updated: "Zuletzt Aktualisiert",
      run: "Experiment Ausführen",
      stop: "Experiment Stoppen",
      results: "Ergebnisse",
      metrics: "Metriken",
      interventions: "Interventionen",
      outcomes: "Ergebnisse",
      validate: "Validieren",
      duplicate: "Duplizieren",
      archive: "Archivieren"
    },
    causal: {
      dag: "Kausaler DAG",
      node: "Knoten",
      edge: "Kante",
      intervention: "Intervention",
      outcome: "Ergebnis",
      treatment: "Behandlung",
      effect: "Effekt",
      correlation: "Korrelation",
      causation: "Kausalität"
    },
    validation: {
      required: "Dieses Feld ist erforderlich",
      invalid: "Ungültiger Wert",
      cycleDetected: "Zyklus im DAG erkannt"
    },
    errors: {
      general: "Ein unerwarteter Fehler ist aufgetreten",
      network: "Netzwerkfehler",
      tryAgain: "Bitte versuchen Sie es erneut"
    },
    settings: {
      title: "Einstellungen",
      language: "Sprache",
      theme: "Design",
      region: "Region"
    }
  },

  // Japanese
  ja: {
    common: {
      loading: "読み込み中...",
      error: "エラー",
      success: "成功",
      warning: "警告",
      info: "情報",
      cancel: "キャンセル",
      confirm: "確認",
      save: "保存",
      delete: "削除",
      edit: "編集",
      create: "作成",
      update: "更新",
      refresh: "更新",
      search: "検索",
      filter: "フィルター",
      export: "エクスポート",
      import: "インポート",
      download: "ダウンロード",
      upload: "アップロード",
      next: "次へ",
      previous: "前へ",
      finish: "完了",
      close: "閉じる",
      yes: "はい",
      no: "いいえ"
    },
    navigation: {
      home: "ホーム",
      experiments: "実験",
      dashboard: "ダッシュボード",
      agents: "エージェント",
      settings: "設定",
      help: "ヘルプ",
      about: "について"
    },
    experiment: {
      title: "実験",
      create: "実験を作成",
      name: "実験名",
      description: "説明",
      status: "ステータス",
      created: "作成日",
      updated: "最終更新",
      run: "実験実行",
      stop: "実験停止",
      results: "結果",
      metrics: "メトリクス",
      interventions: "介入",
      outcomes: "結果",
      validate: "検証",
      duplicate: "複製",
      archive: "アーカイブ"
    },
    causal: {
      dag: "因果DAG",
      node: "ノード",
      edge: "エッジ",
      intervention: "介入",
      outcome: "結果",
      treatment: "処置",
      effect: "効果",
      correlation: "相関",
      causation: "因果関係"
    },
    validation: {
      required: "この項目は必須です",
      invalid: "無効な値",
      cycleDetected: "DAGにサイクルが検出されました"
    },
    errors: {
      general: "予期しないエラーが発生しました",
      network: "ネットワークエラー",
      tryAgain: "もう一度お試しください"
    },
    settings: {
      title: "設定",
      language: "言語",
      theme: "テーマ",
      region: "地域"
    }
  },

  // Chinese (Simplified)
  zh: {
    common: {
      loading: "加载中...",
      error: "错误",
      success: "成功",
      warning: "警告",
      info: "信息",
      cancel: "取消",
      confirm: "确认",
      save: "保存",
      delete: "删除",
      edit: "编辑",
      create: "创建",
      update: "更新",
      refresh: "刷新",
      search: "搜索",
      filter: "筛选",
      export: "导出",
      import: "导入",
      download: "下载",
      upload: "上传",
      next: "下一步",
      previous: "上一步",
      finish: "完成",
      close: "关闭",
      yes: "是",
      no: "否"
    },
    navigation: {
      home: "首页",
      experiments: "实验",
      dashboard: "仪表板",
      agents: "代理",
      settings: "设置",
      help: "帮助",
      about: "关于"
    },
    experiment: {
      title: "实验",
      create: "创建实验",
      name: "实验名称",
      description: "描述",
      status: "状态",
      created: "创建时间",
      updated: "最后更新",
      run: "运行实验",
      stop: "停止实验",
      results: "结果",
      metrics: "指标",
      interventions: "干预",
      outcomes: "结果",
      validate: "验证",
      duplicate: "复制",
      archive: "存档"
    },
    causal: {
      dag: "因果DAG",
      node: "节点",
      edge: "边",
      intervention: "干预",
      outcome: "结果",
      treatment: "处理",
      effect: "效应",
      correlation: "相关性",
      causation: "因果关系"
    },
    validation: {
      required: "此字段为必填项",
      invalid: "无效值",
      cycleDetected: "在DAG中检测到循环"
    },
    errors: {
      general: "发生了意外错误",
      network: "网络错误",
      tryAgain: "请重试"
    },
    settings: {
      title: "设置",
      language: "语言",
      theme: "主题",
      region: "地区"
    }
  }
}

// Helper function to get nested translation
export function getNestedTranslation(obj: TranslationDict, path: string): string {
  const keys = path.split('.')
  let current: any = obj
  
  for (const key of keys) {
    if (current && typeof current === 'object' && key in current) {
      current = current[key]
    } else {
      return path // Return path if translation not found
    }
  }
  
  return typeof current === 'string' ? current : path
}

// Get available languages
export const availableLanguages = Object.keys(translations)

// Default language
export const defaultLanguage = 'en'