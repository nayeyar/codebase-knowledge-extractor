package com.example.sample;

import java.util.ArrayList;
import java.util.List;

public class SampleController {
    @GetMapping("/active")
    public List<String> findActive(List<String> values) {
        List<String> result = new ArrayList<>();
        for (String value : values) {
            if (value != null && !value.isBlank()) {
                result.add(value);
            }
        }
        return result;
    }
}
