import {Component, OnInit, ViewChild, EventEmitter, Input, ElementRef} from '@angular/core';
import {MatSort, MatTableDataSource, MatPaginator} from '@angular/material';
import {Observable, of as observableOf, merge} from 'rxjs';
import {catchError} from 'rxjs/operators';
import {map} from 'rxjs/operators';
import {startWith} from 'rxjs/operators';
import {switchMap} from 'rxjs/operators';
import {Location} from '@angular/common';
import {PaginatorFilter, PaginatorRes} from "./models";
import {BaseService} from "./base.service";

export abstract class Paginator<T> implements OnInit {
    dataSource: MatTableDataSource<T> = new MatTableDataSource();

    @ViewChild(MatPaginator) paginator: MatPaginator;
    @ViewChild(MatSort) sort: MatSort;
    change: EventEmitter<any> = new EventEmitter();

    protected abstract displayed_columns: string[];
    protected default_page_size: number = 15;
    isLoading_results = false;
    total: number;
    private interval: number;


    constructor(
        protected service: BaseService,
        protected location: Location
    ) {

    }

    protected _ngOnInit(){

    }

    get_filter(): any{
        let res = new PaginatorFilter();
        res.page_number = this.paginator.pageIndex;
        res.page_size = this.paginator.pageSize || this.default_page_size;
        res.sort_column = this.sort.active ? this.sort.active : '';
        res.sort_descending = this.sort.direction ? this.sort.direction == 'desc' : true;
        return res;
    }

    ngOnInit() {
        this._ngOnInit();

        // If the user changes the sort order, reset back to the first page.
        this.sort.sortChange.subscribe(() => this.paginator.pageIndex = 0);

        merge(this.sort.sortChange, this.paginator.page, this.change)
            .pipe(
                startWith({}),
                switchMap(() => {
                    this.isLoading_results = true;
                    return this.service.get_paginator<T>(this.get_filter())
                }),
                map(res => {
                    // Flip flag to show that loading has finished.
                    this.isLoading_results = false;

                    return res;
                }),
                catchError(() => {
                    this.isLoading_results = false;
                    return observableOf(new PaginatorRes<T>());
                })
            ).subscribe(res => {
            this.dataSource.data = res.data;
            this.total = res.total
        });

        // this.interval = setInterval(() => this.change.emit('event'), 5000);
    }

    ngOnDestroy() {
        clearInterval(this.interval);
    }

}
