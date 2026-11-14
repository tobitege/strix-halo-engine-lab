    if (!embeds.empty() || m.image_rows_set) {
        const std::size_t embed_width = c.hidden_size;
        std::vector<std::uint8_t> embed_flags(Tn, 0);
        std::vector<std::uint32_t> embed_local(T, UINT32_MAX);
        for (std::uint32_t l = 0; l < Tn; ++l) {
            embed_local[global_row[l]] = l;
            if (!embed_extras.empty()) embed_flags[l] = embed_extras[global_row[l]];
        }
        // A shorter replacement must also erase flags beyond its own rows before declaring the buffer clean.
        if (m.image_rows_set) m.image_rows.zero(st);
        if (Tn != 0U) m.image_rows.upload(embed_flags, st);
        m.image_rows_set = std::find(embed_flags.begin(), embed_flags.end(), std::uint8_t{1}) != embed_flags.end();
        (void)embed_width;
    }
