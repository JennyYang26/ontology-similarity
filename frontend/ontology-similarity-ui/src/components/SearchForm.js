import React, { useState } from 'react';
import axios from 'axios';
import Select from 'react-select';
import SearchFormHTML from './SearchFormHTML';

const algorithmOptions = [
  { value: 'hss', label: '🔍 HSS' },
  { value: 'ratio_hss', label: '📊 Ratio HSS' },
  { value: 'wl', label: '📐 WL + Node2Vec' },
  { value: 'hybrid', label: '🧠 Hybrid SimRank Fusion' }
];

const SearchForm = () => {
  const [targetItem, setTargetItem] = useState('');
  const [algorithm, setAlgorithm] = useState(null);
  const [startIndex, setStartIndex] = useState('');
  const [endIndex, setEndIndex] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [generateGraph, setGenerateGraph] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setResult(null);

    if (!targetItem || !algorithm) {
      setError('Target item and algorithm are required.');
      return;
    }

    const payload = {
      target_item: targetItem,
      algorithm: algorithm.value,
    };

    if (startIndex) payload.start_index = parseInt(startIndex);
    if (endIndex) payload.end_index = parseInt(endIndex);

    try {
      const res = await axios.post(
        `http://localhost:8000/find_similar_items/?generate_graph=${generateGraph}`,
        payload
      );
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Something went wrong');
    }
  };

  return SearchFormHTML({
    targetItem,
    algorithm,
    startIndex,
    endIndex,
    result,
    error,
    handleSubmit,
    setTargetItem,
    setAlgorithm,
    setStartIndex,
    setEndIndex,
    algorithmOptions,
    generateGraph,
    setGenerateGraph
  });
};

export default SearchForm;