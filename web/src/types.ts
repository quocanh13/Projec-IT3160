export interface Symptom{
    /**Vietnamese name of the symptom*/
    vname: string,

    /**English name of the symptom*/
    ename: string,

    /**Whether the patient has this symptom*/
    hasSymptom: boolean 
}

export interface Disease {
  /** Vietnamese name of the disease */
  vname: string

  /** English name of the disease */
  ename: string

  /** Risk percentage predicted by the model */
  riskPercent: number
}