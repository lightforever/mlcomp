import {Component, ViewChild} from '@angular/core';
import {Paginator} from "../../paginator";
import {Space, SpaceFilter} from "../../models";
import {Location} from "@angular/common";
import {MatDialog} from "@angular/material/dialog";
import {SpaceService} from "./space.service";
import {Helpers} from "../../helpers";
import {SpaceAddDialogComponent} from "./space-add-dialog";
import {MatTableDataSource} from "@angular/material/table";
import {MatPaginator} from "@angular/material/paginator";
import {MatSort} from "@angular/material/sort";
import {SpaceRunDialogComponent} from "./space-run-dialog";
import {COMMA, ENTER} from "@angular/cdk/keycodes";
import {MatChipInputEvent} from "@angular/material/chips";
import {MatAutocompleteSelectedEvent} from "@angular/material/autocomplete";
import {MatIconRegistry} from "@angular/material/icon";
import {DomSanitizer} from "@angular/platform-browser";

@Component({
    selector: 'app-space',
    templateUrl: './space.component.html',
    styleUrls: ['./space.component.css']
})
export class SpaceComponent extends Paginator<Space> {
    displayed_columns: string[] = [
        'name',
        'created',
        'changed',
        'tags'
    ];

    filter_hidden: boolean = true;
    filter_applied_text: string = '';

    name: string;
    selected: Space;

    relation_dataSource: MatTableDataSource<Space> = new MatTableDataSource();
    relation_total: number = 0;
    @ViewChild('relation_paginator') relation_paginator: MatPaginator;
    @ViewChild('relation_sort') relation_sort: MatSort;
    relation_selected: Space;
    relation_filter_hidden: boolean = true;
    relation_filter_applied_text: string = '';
    relation_name: string;

    separatorKeysCodes: number[] = [ENTER, COMMA];
    tags: string[] = [];
    chosen_spaces = [];
    names: string[] = [];
    filter_tags: string[] = [];
    filter_all_tags: string[] = [];

    filter_tags_related: string[] = [];
    filter_all_tags_related: string[] = [];

    constructor(
        protected service: SpaceService,
        protected location: Location,
        public dialog: MatDialog,
        iconRegistry: MatIconRegistry,
        sanitizer: DomSanitizer
    ) {
        super(service, location);
        this.id_column = 'name';

        iconRegistry.addSvgIcon('delete',
            sanitizer.bypassSecurityTrustResourceUrl(
                'assets/img/delete.svg'));
        iconRegistry.addSvgIcon('pin',
            sanitizer.bypassSecurityTrustResourceUrl(
                'assets/img/pin.svg'));
    }

    get chosen_spaces_or() {
        let res = [];
        for (let s of this.chosen_spaces) {
            if (s.type == 'tmp' || s.logic == 'or') {
                res.push(s);
            }
        }
        return res;
    }

    get chosen_spaces_and() {
        let res = [];
        for (let s of this.chosen_spaces) {
            if (s.type == 'tmp' || s.logic == 'and') {
                res.push(s);
            }
        }
        return res;
    }

    filter_remove_tag(tag) {
        let index = this.filter_tags.indexOf(tag);
        this.filter_tags.splice(index, 1);
        this.change.emit();
    }

    filter_remove_tag_related(tag) {
        let index = this.filter_tags_related.indexOf(tag);
        this.filter_tags_related.splice(index, 1);
        this.relation_changed();
    }

    filter_tag_add(event: MatChipInputEvent) {
        const input = event.input;
        let value = event.value;

        // Add our fruit
        if ((value || '').trim()) {
            value = value.trim();
            this.filter_tags.push(value);
        }

        // Reset the input value
        if (input) {
            input.value = '';
        }
        this.change.emit();
    }

    filter_tag_add_related(event: MatChipInputEvent) {
        const input = event.input;
        let value = event.value;

        // Add our fruit
        if ((value || '').trim()) {
            value = value.trim();
            this.filter_tags_related.push(value);
        }

        // Reset the input value
        if (input) {
            input.value = '';
        }
        this.relation_changed();
    }

    filter_tag_selected(event: MatAutocompleteSelectedEvent) {
        this.filter_tags.push(event.option.viewValue);
        this.change.emit();
    }

    filter_tag_selected_related(event: MatAutocompleteSelectedEvent) {
        this.filter_tags_related.push(event.option.viewValue);
        this.relation_changed();
    }

    chosen_remove_space(space) {
        let index = this.chosen_spaces.indexOf(space);
        this.chosen_spaces.splice(index, 1);
    }

    chosen_fix_space(logic) {
        for (let space of this.chosen_spaces) {
            if (space.type == 'tmp') {
                space.type = 'const';
                space.logic = logic;
            }
        }
    }

    chosen_space_add(event: MatChipInputEvent) {
        const input = event.input;
        let value = event.value;

        if ((value || '').trim()) {
            value = value.trim();
            this.chosen_spaces.push({
                'value': value,
                'type': 'const',
                'logic': 'or'
            });
        }

        if (input) {
            input.value = '';
        }
    }

    chosen_space_selected(event: MatAutocompleteSelectedEvent) {
        this.chosen_spaces.push({
            'value': event.option.viewValue,
            'type': 'const',
            'logic': 'or'
        });
    }

    onchange() {
        this.change.emit();
        let count = 0;
        if (this.name) count += 1;
        this.filter_applied_text = count > 0 ? `(${count} applied)` : '';
    }

    get_filter(): any {
        let res = new SpaceFilter();
        res.paginator = super.get_filter();
        res.name = this.name;
        res.tags = this.filter_tags;
        return res;
    }

    protected _ngOnInit() {
        super._ngOnInit();
        this.relation_paginator.page.subscribe(x => {
            this.relation_changed()
        });
        this.relation_sort.sortChange.subscribe(x => {
            this.relation_changed()
        });
    }

    update_tags(event = null) {
        let name = '';
        if (event) {
            name = event.target.value;
        }

        this.service.tags({'name': name}).subscribe(x => {
            this.tags = x.tags;
        })
    }

    update_names(event = null) {
        let name = '';
        if (event) {
            name = event.target.value;
        }

        this.service.names({'name': name}).subscribe(x => {
            this.names = x.names;
        })
    }

    update_filter_all_tags(event = null) {
        let name = '';
        if (event) {
            name = event.target.value;
        }

        this.service.tags({'name': name}).subscribe(x => {
            this.filter_all_tags = x.tags;
        })
    }

    update_filter_all_tags_related(event = null) {
        let name = '';
        if (event) {
            name = event.target.value;
        }

        this.service.tags({'name': name}).subscribe(x => {
            this.filter_all_tags_related = x.tags;
        })
    }

    run() {
        this.dialog.open(SpaceRunDialogComponent, {
            width: '2000px', height: '900px',
            data: {'spaces': this.chosen_spaces}
        });
    }

    add() {
        const dialogRef = this.dialog.open(SpaceAddDialogComponent, {
            width: '1000px', height: '700px',
            data: {method: 'add', space: {'name': ''}}
        });

        dialogRef.afterClosed().subscribe(result => {
            this.change.emit();
        });
    }

    edit() {
        const dialogRef = this.dialog.open(SpaceAddDialogComponent, {
            width: '1000px', height: '700px',
            data: {method: 'edit', space: Helpers.clone(this.selected)}
        });

        dialogRef.afterClosed().subscribe(result => {
            this.change.emit();
        });
    }

    copy() {
        const dialogRef = this.dialog.open(SpaceAddDialogComponent, {
            width: '800px', height: '700px',
            data: {
                method: 'copy',
                space: Helpers.clone(this.selected),
                'old_space': this.selected.name
            }
        });

        dialogRef.afterClosed().subscribe(result => {
            this.change.emit();
        });
    }

    remove() {
        this.service.remove(this.selected.name).subscribe(result => {
            this.selected = null;
            this.relation_selected = null;
            this.relation_dataSource.data = [];
            this.relation_paginator.pageIndex = 0;
            this.change.emit();
        });
    }

    relation_changed() {
        if (!this.selected) {
            this.relation_dataSource.data = [];
            this.relation_total = 0;
            return;
        }
        let filter = {
            'parent': this.selected.name,
            'paginator': {
                'page_number': this.relation_paginator.pageIndex,
                'page_size': this.relation_paginator.pageSize,
                'sort_column': this.relation_sort.active ? this.relation_sort.active : '',
                'sort_descending': this.relation_sort.direction ?
                    this.relation_sort.direction == 'desc' : true
            },
            'name': this.relation_name,
            'tags': this.filter_tags_related
        };
        this.service.get_paginator<Space>(filter).subscribe(res => {
            this.relation_dataSource.data = res.data;
            this.relation_total = res.total;
        })
    }

    onSelected(row: Space, event: MouseEvent) {
        let spaces = [row];
        if (event.shiftKey && this.selected) {
            let selected_index = this.dataSource.data.indexOf(this.selected);
            let row_index = this.dataSource.data.indexOf(row);

            if (selected_index >= 0 && row_index >= 0) {
                for (let i = selected_index + 1; i < row_index + 1; i++) {
                    spaces.push(this.dataSource.data[i]);
                }
            }

            event.stopPropagation();
        }

        for (let space of spaces) {
            if (this.chosen_spaces.length > 0 && !event.ctrlKey && !event.shiftKey) {
                let last_index = this.chosen_spaces.length - 1;
                if (this.chosen_spaces[last_index].type == 'tmp') {
                    this.chosen_spaces.splice(last_index, 1);
                }
            }
            let found = false;
            for (let s of this.chosen_spaces) {
                if (s.value == space.name) {
                    found = true;
                    break
                }
            }

            if (!found) {
                this.chosen_spaces.push({
                    'value': space.name,
                    'type': 'tmp',
                    'logic': 'or'
                });
            }

            this.selected = space;
            this.relation_selected = null;
            this.relation_paginator.pageIndex = 0;

        }
        if (event.ctrlKey || event.shiftKey) {
            this.selected = null;
        }
        this.relation_changed();
    }

    relation_append() {
        this.service.relation_append(this.selected.name, this.relation_selected.name).subscribe(res => {
            this.relation_changed()
        });
    }

    relation_remove() {
        this.service.relation_remove(this.selected.name, this.relation_selected.name).subscribe(res => {
            this.relation_changed()
        });
    }

    remove_tag(space: Space, tag: string) {
        this.service.tag_remove(space.name, tag).subscribe(res => {
            this.change.emit();
        });
        this.relation_changed();
    }

    tag_add(space: Space, event: MatChipInputEvent) {
        const input = event.input;
        let value = event.value;

        // Add our fruit
        if ((value || '').trim()) {
            value = value.trim();
            space.tags.push(value);
            this.service.tag_add(space.name, value).subscribe(res => {
            });
        }

        // Reset the input value
        if (input) {
            input.value = '';
        }

        this.relation_changed();
    }

    tag_selected(space: Space, event: MatAutocompleteSelectedEvent) {
        this.service.tag_add(space.name, event.option.viewValue).subscribe(res => {
        });
        space.tags.push(event.option.viewValue);
        this.relation_changed();
    }
}
