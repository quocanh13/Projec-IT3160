import type { Disease, Symptom } from './types'
import { useSymptomStore } from './store/SymptomStore'

import "./styles/DiseasePrediction.css"
import "./styles/SymptomSection.css"
import "./styles/DiseaseSection.css"
import searchIcon from "./assets/icon/search.png" 
import type { ChangeEvent } from 'react'

const symptomList: Symptom[] = [
    {ename : "anxiety and nervousness", vname : "lo âu và bồn chồn", hasSymptom: false},
    {ename : "depression", vname : "trầm cảm", hasSymptom: true},
    {ename : "shortness of breath", vname : "khó thở", hasSymptom: false},
    {ename : "depressive or psychotic symptoms", vname : "triệu chứng trầm cảm hoặc loạn thần", hasSymptom: true},
    {ename : "sharp chest pain", vname : "đau ngực nhói", hasSymptom: false},
    {ename : "dizziness", vname: "chóng mặt", hasSymptom: true},
    {ename : "insomnia", vname: "mất ngủ", hasSymptom: true}
]
const diseaseList: Disease[] = [
    {ename : 'panic disorder', vname : 'rối loạn hoảng sợ', riskPercent: 0.68},
    {ename : 'vocal cord polyp', vname : 'polyp dây thanh quản', riskPercent: 0.0},
    {ename : 'turner syndrome', vname : 'hội chứng turner', riskPercent: 0.99},
    {ename : 'cryptorchidism', vname : 'tinh hoàn ẩn', riskPercent: 0.0},
    {ename : 'fracture of the hand', vname : 'gãy xương bàn tay', riskPercent: 0.36},
    {ename : 'cellulitis or abscess of mouth', vname : 'viêm mô tế bào hoặc áp xe miệng', riskPercent: 0.76}
]

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
    const {setSymptomSearchTerm} = useSymptomStore()
    function onSearchTermChange(e: ChangeEvent<HTMLInputElement>){
        setSymptomSearchTerm(e.target.value)
    }

    return (
        <div className='symptom-search-bar-container'>
            <form action="" className='symptom-search-bar-form'>
                <img src={searchIcon} alt="" className='symptom-search-icon'/>
                <input type="text" className='symptom-search-input' onChange={onSearchTermChange}/>
            </form>
        </div>
    )
}

function SymptomList(){
    const {symptomSearchTerm} = useSymptomStore()
    return (
        <div className='symptom-list-container'>
            {
            symptomList
                .filter((v)=>{return v.vname.includes(symptomSearchTerm)})
                .sort((a, b) => a.vname.localeCompare(b.vname))
                .map(s=>{return <SymptomItem symptom={s} key={s.ename}></SymptomItem>})
            }
        </div>
    )
}

function SymptomItem({symptom} : {symptom : Symptom}){
    return (
        <div className={'symptom-item-container ' + (symptom.hasSymptom ? "active" : "")}>
            <div className='symptom-item'>
                <p className='symptom-vname'>{symptom.vname}</p>
                <p className='symptom-ename'>{symptom.ename}</p>
            </div>
        </div>
    )
}

function DiseaseSection(){
    return (
        <div className='disease-container'>
            <div className='disease-predict-button-container'>
                <button className='disease-predict-button'>Predict Disease</button>
            </div>
            <DiseaseList></DiseaseList>
        </div>
    )
}

function DiseaseList() {
    return (
        <div className='disease-list-container'>
            {
            diseaseList
                .sort((a, b) => b.riskPercent - a.riskPercent)
                .map(d => (
                    <DiseaseItem disease={d} key={d.ename} />
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