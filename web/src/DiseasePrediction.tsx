import type { Disease, Symptom } from './types'
import { useSymptomStore } from './store/symptomStore'
import {softmax_predict} from "./api/predict"

import "./styles/DiseasePrediction.css"
import "./styles/SymptomSection.css"
import "./styles/DiseaseSection.css"
import searchIcon from "./assets/icon/search.png" 
import type { ChangeEvent } from 'react'
import { useDiseaseStore } from './store/diseaseStore'

export function DiseasePrediction() {
  return (
    <div className='main-container'>
        <h2>Disease Prediction</h2>
        <div className='disease-prediction-container'>
            <SymptomSection></SymptomSection>
            <DiseaseSection></DiseaseSection>
        </div>
    </div>
  )
}

function SymptomSection(){
    return (
        <div className='symptom-container'>
            <SymptomSearchBar></SymptomSearchBar>
            <SymptomList></SymptomList>
        </div>
    )
}

function SymptomSearchBar(){
    const {setSymptomSearchTerm, resetSymptom} = useSymptomStore()
    function onSearchTermChange(e: ChangeEvent<HTMLInputElement>){
        setSymptomSearchTerm(e.target.value)
    }

    return (
        <div className='symptom-search-bar-container'>
            <form action="" className='symptom-search-bar-form'>
                <img src={searchIcon} alt="" className='symptom-search-icon'/>
                <input type="text" className='symptom-search-input' onChange={onSearchTermChange}/>
            </form>
            <button className='symptom-reset-button' onClick={resetSymptom}>Reset</button>
        </div>
    )
}

function SymptomList(){
    const {symptomList} = useSymptomStore()
    const {symptomSearchTerm} = useSymptomStore()
    return (
        <div className='symptom-list-container'>
            {
            symptomList
                .filter((v)=>{return v.vname.toLowerCase().includes(symptomSearchTerm.toLowerCase())})
                .sort((a, b) => a.vname.localeCompare(b.vname))
                .map(s=>{return <SymptomItem symptom={s} key={s.id}></SymptomItem>})
            }
        </div>
    )
}

function SymptomItem({symptom} : {symptom : Symptom}){
    const {toggleSymptom} = useSymptomStore()
    function onClickSymptom(){
        toggleSymptom(symptom.id)
    }

    return (
        <div className={'symptom-item-container ' + (symptom.hasSymptom ? "active" : "")} onClick={onClickSymptom}>
            <div className='symptom-item'>
                <p className='symptom-vname'>{symptom.vname.toLocaleUpperCase()}</p>
                <p className='symptom-ename'>{symptom.ename}</p>
            </div>
        </div>
    )
}

function DiseaseSection(){
    const {getHasSymptomList} = useSymptomStore()
    const {setRiskPercent} = useDiseaseStore()
    async function onClickPredict(){
        const symptomList = getHasSymptomList()
        const res = await softmax_predict(symptomList)
        console.log(res)
        setRiskPercent(res)
    }

    return (
        <div className='disease-container'>
            <div className='disease-predict-button-container'>
                <button className='disease-predict-button' onClick={onClickPredict}>Predict Disease</button>
            </div>
            <DiseaseList></DiseaseList>
        </div>
    )
}

function DiseaseList() {
    const {diseaseList} = useDiseaseStore()
    return (
        <div className='disease-list-container'>
            {
            [...diseaseList]
                .sort((a, b) => b.riskPercent - a.riskPercent)
                .map(d => (
                    <DiseaseItem disease={d} key={d.id} />
                ))}
        </div>
    )
}

function DiseaseItem({disease} : {disease: Disease}) {
    return (
        <div className='disease-item-container'>
            <div className='disease-item'>
                <div className='disease-name-container'>
                    <p className='disease-vname'>{disease.vname}</p>
                    <p className='disease-ename'>{disease.ename}</p>
                </div>
                <div className='disease-percent-risk-container'>
                    <p className='disease-percent-risk'>{disease.riskPercent}</p>
                </div>
            </div>
        </div>
    )
}