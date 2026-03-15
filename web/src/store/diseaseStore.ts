import { create } from "zustand"
import diseaseList from "../assets/data/disease_data.json" 
import type { Disease } from "../types"

interface DiseaseStore{
    diseaseList : Disease[],
    setRiskPercent: (riskList: number[]) => void
}

export const useDiseaseStore = create<DiseaseStore>((set) => ({
    diseaseList,
    setRiskPercent(riskList) {
        set(state => ({
            diseaseList : state.diseaseList.map((v, i) => ({...v, riskPercent : riskList[i]}))   
        }))
    },
}))